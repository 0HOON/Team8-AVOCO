import streamlit as st
from dotenv import load_dotenv

from htmlTemplates import css
from dotenv import load_dotenv

from utils import prepare_chain, instructions
load_dotenv()

## TODO review 전부 가져오기
## TODO 결과 복사 버튼 만들기

def reset():
  if st.button("reset"):
    st.session_state.clear()
    st.rerun()

def submit():
  url = st.session_state.url_input
  st.session_state.clear()
  st.session_state.chain = prepare_chain(url)
  st.session_state.url_input = ''

def main():
  # load env settings
  load_dotenv()

  # session_state list
  # - url: OpenReview URL input
  # - title: title of the paper
  # - chain: langchain object
  # - review/inconsistency/discussion_summary: generated summary text

  st.set_page_config(page_title="Meta Review Helper", page_icon=":books:", layout="wide")

  st.write(css, unsafe_allow_html=True)
  
  with st.sidebar:
    st.header("Meta Review Helper :books:")
    st.text_input("Enter OpenReview URL of the paper:", key='url_input', on_change=submit)
        
  page_dict = {}
  if "title" in st.session_state:
    st.title(st.session_state.title)

  review_summary_page = st.Page("st_pages/reviewSummary.py", title="Review Summary", icon=":material/handyman:", default=True)
  inconsitency_summary_page = st.Page("st_pages/inconsistencySummary.py", title="Inconsistency Summary", icon=":material/handyman:")
  discussion_summary_page = st.Page("st_pages/discussionSummary.py", title="Discussion Summary", icon=":material/handyman:")

  pages = [review_summary_page, inconsitency_summary_page, discussion_summary_page]
  page_dict["Main Functions"] = pages

  reset_page = st.Page(reset, title="reset", icon=":material/logout:")
  pg = st.navigation({"Settings": [reset_page]} | page_dict)
  pg.run()        

  

if __name__ == '__main__':
  main()
