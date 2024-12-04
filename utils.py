import io
import regex as re
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
  "inconsistency_summary": "Find any inconsistency between reviewers and summarize it in a form like '**Inconsistency 1**: ..., **Inconsistency 2**: ...'. Each summary should be about 5 sentences. Then recommend which page of the paper should I read to effectively resolve the inconsistency. Recommendation should be like '**Important Pages**: ...'. If there is no inconsistency at all, just answer no.",
  "discussion_summary": "Summarize each discussion a form like '**Discussion 1**: ..., **Discussion 2**: ...'. Each summary should be about 5 sentences. Here 'discussion' means a review and the replies for that review. Each summary should list brief summary of the review, main points of the discussion, and whether the reply of authors was appropriate.",
  "find_inconsistency_in_pdf": "You are a research assistant tasked with identifying the exact text from a research paper corresponding to a specific description of its content. You will be provided with two inputs: Paper Text: The full text of the paper (paper_text). \
    Inconsistency Text: A description of a section or part of the paper where inconsistencies in reviews are observed (inconsistency_text). \
      Your goal is to: Find the exact text from the paper_text that matches or is most relevant to the inconsistency_text. Return only the relevant text as it appears in the paper_text without any additional explanations or modifications. Please exclude any mathematical expressions, and make sure to extract the exact text from the paper text without any modifications. \
        Returns form like '{**'Inconsistency 1'**: ..., **'Inconsistency 2'**: ...} "
} 

tokens = {
  "review_summary": ['review', 'review summary', 'reviewer', 'strength', 'weakness', 'keywords', 'important pages'],
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

  def get_text(self, level):
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

    for rep in self.replies:
      text += rep.get_text(level+1)
      
    return text
  

def get_reviews_and_pdf_from_url(url, venue_id="ICLR.cc/2023/Conference"):
  client = openreview.Client()
  paper_id = url.split('=')[-1]
  paper_info = client.get_note(paper_id)
  #st.write(f"Title: {paper_info.content['title']}")

  st.session_state.title = paper_info.content['title']
  st.session_state.authors = paper_info.content['authors']
  st.session_state.keywords = paper_info.content['keywords']
  st.session_state.abstract = paper_info.content['abstract']

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
  return root.get_text(0)

def get_pdf_text(pdf):
  text = ""
  reader = PdfReader(io.BytesIO(pdf))
  for page in reader.pages:
      text += page.extract_text()
  return text

def get_texts_from_url(url):
  reviews, pdf = get_reviews_and_pdf_from_url(url)

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

def prepare_chain(url):
  with st.spinner("Collecting Paper Data"):
    review_text, pdf_text,pdf = get_texts_from_url(url)
    st.session_state.full_text = review_text
    st.session_state.paper_text = pdf_text
    st.session_state.paper_pdf=pdf
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

def represent_pdf(search_strings):
  pdf_document=fitz.open("pdf",st.session_state.paper_pdf)
  for text in search_strings:
      for page_num in range(len(pdf_document)):
          page = pdf_document[page_num]
          # 텍스트 검색 (출력 형식: [(x0, y0, x1, y1, "찾은 텍스트", idx), ...])
          text_instances = page.search_for(text)
          
          if text_instances:
              # 각 검색 결과에 대해 사각형 하이라이트 추가
              for rect in text_instances:
                  highlight = page.add_highlight_annot(rect)
                  highlight.update()  # 업데이트 적용

  # 하이라이트된 PDF 저장
  output_filename = f"{st.session_state.title}_find_inconsistency.pdf"
  pdf_document.save(output_filename)
  pdf_document.close()
  return output_filename
  
    