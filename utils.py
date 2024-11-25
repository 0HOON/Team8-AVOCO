import io
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


instructions = {
  "review_summary": "Summarize each reviews in a form like '**Review 1**: ..., **Review 2**: ...'. Each summary should be about 3 sentences. Then, based on the reviews, recommend which pages of the paper should I read to effectively understand the opinions of the reviewers. Recommendation should be like '**Important Pages**: ...'",
  "inconsistency_summary": "Find any inconsistency between reviewers and summarize it in a form like '**Inconsistency 1**: ..., **Inconsistency 2**: ...'. Each summary should be about 5 sentences. Then recommend which page of the paper should I read to effectively resolve the inconsistency. Recommendation should be like '**Important Pages**: ...'. If there is no inconsistency at all, just answer no.",
  "discussion_summary": "Summarize each discussion a form like '**Discussion 1**: ..., **Discussion 2**: ...'. Each summary should be about 5 sentences. Here 'discussion' means a review and the replies for that review. Each summary should list brief summary of the review, main points of the discussion, and whether the reply of authors was appropriate."
} 

def get_reviews_and_pdf_from_url(url, venue_id="ICLR.cc/2023/Conference"):
  client = openreview.Client()
  paper_id = url.split('=')[-1]
  paper_info = client.get_note(paper_id)
  #st.write(f"Title: {paper_info.content['title']}")

  st.session_state.title = paper_info.content['title']
  st.session_state.authors = paper_info.content['authors']
  st.session_state.keywords = paper_info.content['keywords']
  st.session_state.abstract = paper_info.content['abstract']

  reviews = client.get_notes(replyto=paper_id, details='replies')
  pdf = client.get_pdf(paper_id)
  return reviews, pdf
    
def get_reviews_text(reviews):
  text = ""
  reviews = [review for review in reviews if 'title' not in review.content]
  for i, review in enumerate(reviews):
    text += f"\n\n\n#### Review {i+1} ####\n"
    text += "\n\n# Summary of the paper\n"
    text += review.content['summary_of_the_paper']
    text += "\n\n# Strength and Weaknesses\n"
    text += review.content['strength_and_weaknesses']
    text += "\n\n# Clarity, Quality, Novelty And Reproducibility\n"
    text += review.content['clarity,_quality,_novelty_and_reproducibility']
    text += "\n\n# Summary of the review\n"
    text += review.content['summary_of_the_review']
    text += "\n\n"
    for j, reply in enumerate(review.details['replies']):
      text += f"\n## Reply {j+1}\n"
      text += reply['content']['comment']
      
  return text

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

  return reviews_text, pdf_text

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
    review_text, pdf_text = get_texts_from_url(url)
    st.session_state.full_text = review_text    
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