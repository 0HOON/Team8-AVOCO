import io
import regex as re
import pickle
import os
import streamlit as st

from PyPDF2 import PdfReader

from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain.text_splitter import CharacterTextSplitter

from langchain_community.vectorstores import FAISS

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

import openreview

import fitz


instructions = {
  "review_summary": '''
  Identify all the reviews from reviewers and summarize them using the following format for each review:

  <review>Review number</reveiw>
  <review summary> Summary of the review in about 5 sentences </review summary>
  <reviewer>Reviewer ID</reviewer>
  <strength> Strengths of the paper highlighted by the reviewer </strength>
  <weakness> Weakness of the paper highlighted by the reviewer </weakness>
  <keywords> Key words of the review </keywords>
  For each review, recommend which pages of the paper should I read to effectively address it. Use the format: <important pages>Important Pages numbers and contents</important pages>

  Include the special tokens in your response as well.
  '''
,
"inconsistency_summary": '''Identify any inconsistencies among reviewers and summarize them using the following format exactly for each inconsistency:

    <inconsistency>Inconsistency number<\inconsistency>
    <inconsistency summary> Summary of the inconsistency in about 5 sentences <\inconsistency summary>
    <Reviewer>Reviewer ID<\Reviewer> <Comments>Reviewer's comments related to this inconsistency<\Comments>

    For each inconsistency, recommend which pages of the paper should be reviewed to effectively address it. Use the format: <important pages>Important Pages numbers and contents<\important pages>

    For each inconsistency, all related reveiwers and thier comments are included with each special tags (include closing token).
    Include the special tokens in your response as well. To handle the closing token correctly, here's the updated explanation and implementation with precise usage of backslashes. Make sure to match the pairs precisely because they will be parsed using special tokens. Always return the results exactly in the format I specify.  ''',
              # If there is no inconsistency at all, simply respond with \
              # No inconsistencies found. Include this exact text in your response.",
  "discussion_summary": '''
  From given review and replies, identify the key aspects of the discussion using the following format:

  <discussion>Discussion with "Reviewer ID"</discussion>
  <key issues> Main concerns or questions raised by the reviewer </key issues>
  <response> Summary about how did the author address the key issues, including any evidence or reasoning they provided </response>
  <evaluation> Evaluate whether the response was adequate or if there were gaps or areas for improvement </evaluation>
  <reviewer reaction> Summary about how did the reviewer assess the author's response and if the reviewer changes their opinion or score as a result <reviewer reaction>

  For each review, recommend which pages of the paper should I read to effectively address it. Use the format: <important pages>Important Pages numbers and contents</important pages>

  Include the special tokens in your response as well. And underline important key words of your response.
  ''',

  "find_inconsistency_in_pdf": "You are a research assistant tasked with identifying the exact text from a research paper corresponding to a specific description of its content. You will be provided with two inputs: Paper Text: The full text of the paper (paper_text). \
    Inconsistency Text: A description of a section or part of the paper where inconsistencies in reviews are observed (inconsistency_text). \
      Your goal is to: Find the exact text from the paper_text that matches or is most relevant to the inconsistency_text. Return only the relevant text as it appears in the paper_text without any additional explanations or modifications. Please exclude any mathematical expressions, and make sure to extract the exact text from the paper text without any modifications. \
        Returns form like '{**'Inconsistency 1'**: ..., **'Inconsistency 2'**: ...} "
} 

tokens = {
  "review_summary": ["review", "review summary", "reviewer", "strength", "weakness", "important pages"],
  "discussion_summary": ["discussion", "key issues", "response", "evaluation", "reviewer reaction", "important pages"]
}

def parse_text(text, tokens):
  parsed_data = {}
  for token in tokens:
    # Regular expression to extract content between the token tags
    pattern = f"<{token}>(.*?)<\\/{token}>"
    matches = re.findall(pattern, text, re.DOTALL)  # DOTALL to include newlines
    parsed_data[token] = matches

  return parsed_data

class NoteNode:
  def __init__(self, note):
    '''
      TreeNode for openreview Notes
    '''
    self.id = note.id
    self.title = note.content.get('title')
    self.writer = note.writers[-1].split('/')[-1]
    self.content = note.content
    self.replyto = note.replyto
    self.replies = []
  
  def add_reply(self, node):
    if self.id == node.replyto:
      self.replies.append(node)
      node.replyto = self
      return True

    if len(self.replies) > 0:
      for rep in self.replies:
        if rep.add_reply(node):
          return True
      return False
    else:
      return False

  def get_text(self, level, recursive=True):
    text = ''
    if self.title: # reply
      if self.replyto is not None: # else root
        text += f"\n{'#'*level} {self.writer}'s reply to {self.replyto.writer}\n"
        text += f"title: {self.title}\n"
        text += f"comment: {self.content['comment']}\n"    
    else: # review
      text += f"\n\n\n# Review from {self.writer}\n"
      text += "\n\n## Summary of the paper\n"
      text += self.content['summary_of_the_paper']
      text += "\n\n## Strength and Weaknesses\n"
      text += self.content['strength_and_weaknesses']
      text += "\n\n## Clarity, Quality, Novelty And Reproducibility\n"
      text += self.content['clarity,_quality,_novelty_and_reproducibility']
      text += "\n\n## Summary of the review\n"
      text += self.content['summary_of_the_review']
      text += "\n\n"

    if recursive:
      for rep in self.replies:
        text += rep.get_text(level+1)
      
    return text
  

def get_reviews_and_pdf(note:openreview.Note, venue_id="ICLR.cc/2023/Conference"):
  client = openreview.Client()
  paper_id = note.id

  st.session_state.title = note.content['title']
  st.session_state.authors = note.content['authors']
  st.session_state.keywords = note.content['keywords']
  st.session_state.abstract = note.content['abstract']

  reviews = sorted(client.get_all_notes(forum=paper_id), key=lambda x: x.cdate)
  pdf = client.get_pdf(paper_id)
  return reviews, pdf
    
def get_reviews_text(reviews):
  root = NoteNode(reviews[0])
  for note in reviews[1:]:
    if note.content.get('title') and 'Decision'in note.content['title']:
      break
    node = NoteNode(note)
    root.add_reply(node)
  st.session_state.root = root
  return root.get_text(0)

def get_pdf_text(pdf):
  text = ""
  reader = PdfReader(io.BytesIO(pdf))
  for page in reader.pages:
      text += page.extract_text()
  return text

def get_texts(note:openreview.Note):
  reviews, pdf = get_reviews_and_pdf(note)

  reviews_text = get_reviews_text(reviews)
  pdf_text = get_pdf_text(pdf)

  return reviews_text, pdf_text, pdf

def get_text_chunks(raw_text):
  splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=400,
    length_function=len,
  )
  chunks = splitter.split_text(raw_text)
  return chunks

def get_vectorstore(text_chunks):
  embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
  #embeddings = HuggingFaceEmbeddings(model_name="dunzhang/stella_en_1.5B_v5", model_kwargs={'device':'mps'})
  vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
  return vectorstore

def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)

def prepare_chain(note:openreview.Note):
  with st.spinner("Collecting Paper Data"):
    review_text, pdf_text, pdf = get_texts(note)
    st.session_state.full_text = review_text
    st.session_state.paper_text = pdf_text
    st.session_state.paper_pdf= pdf
    text_chunks = get_text_chunks(pdf_text)
  with st.spinner("Building Vector Store"):
    vectorstore = get_vectorstore(text_chunks)
  with st.spinner("Constructing LangChain"):
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    parser = StrOutputParser()

    prompt = PromptTemplate.from_template(
      '''You are a meta reviewer of ICLR 2023 Conference. Follow the instruction using following pieces of retrieved context of academic paper. If you don't know the answer, just say that you don't know. Keep the answer concise.
      Instruction: {instruction}

      Context: {context} 

      Answer:'''
    )

    prompt_chain = (
      {"instruction": RunnablePassthrough() , "context": vectorstore.as_retriever(k=10, fetch_k=30) | format_docs}
      | prompt
      | llm
      | parser
    )

  return prompt_chain

def represent_pdf(search_strings, reviewers):

    pdf_document = fitz.open("pdf", st.session_state.paper_pdf)
    
    for inconsistency_text in search_strings:
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            # Search for the inconsistency text in the document
            text_instances = page.search_for(inconsistency_text)
            
            if text_instances:
                # Add highlights and reviewer IDs as text annotations
                offset=0
                for rect in text_instances:
                    # Add highlight annotation
                    highlight = page.add_highlight_annot(rect)
                    highlight.update()  # Apply the highlight

                    # Convert rect to (x, y) point for placing the reviewer ID text
                    for id in reviewers:
                      text_position = (page.rect.x0 + 10, rect.y0+(offset))  # Left margin, aligned with the top of the highlight

                      # Add reviewer ID as visible text
                      page.insert_text(
                          text_position,
                          f"{id}",
                          fontsize=10,  # Font size
                          color=(0, 0, 1),  # Text color (black)
                          fontname="helv",  # Font name
                      )
                      offset+=10

    # Save the modified PDF
    output_filename = f"{st.session_state.title}_find_inconsistency_with_reviewers.pdf"
    pdf_document.save(output_filename)
    pdf_document.close()
    return output_filename

def get_paper_list():
  file_path = "./paper_list.pkl"
  if os.path.isfile(file_path):
    with open(file_path, 'rb') as f:
      papers = pickle.load(f)
  else:
    client = openreview.Client()
    papers = client.get_all_notes(signature='ICLR.cc/2023/Conference')
    with open(file_path, 'wb') as f:
      pickle.dump(papers, f)
  return papers
  