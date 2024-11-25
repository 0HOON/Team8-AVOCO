import streamlit as st
from dotenv import load_dotenv

from htmlTemplates import css
from dotenv import load_dotenv

from utils import prepare_chain, instructions
load_dotenv()

## TODO review 전부 가져오기
## TODO 결과 복사 버튼 만들기

def get_url():
  st.header("Meta Review Helper :books:")
  url = st.text_input("Enter OpenReview URL of the paper:")
  if st.button("Summarize!"):
    st.session_state.chain = prepare_chain(url)
    st.rerun()

def reset():
  st.session_state.clear()
  st.rerun()

def main():
  # load env settings
  load_dotenv()

  # session_state list
  # - url: OpenReview URL input
  # - title: title of the paper
  # - chain: langchain object
  # - review/inconsistency/discussion_summary: generated summary text

  st.set_page_config(page_title="Meta Review Helper", page_icon=":books:", layout="wide")
  
  #st.write(css, unsafe_allow_html=True)

  dashboard_page = st.Page("st_pages/dashboard.py", title="Dashboard", icon=":material/browse_activity:", default=True)
  review_summary_page = st.Page("st_pages/reviewSummary.py", title="Review Summary", icon=":material/emoji_people:")
  inconsitency_summary_page = st.Page("st_pages/inconsistencySummary.py", title="Inconsistency Summary", icon=":material/connect_without_contact:")
  discussion_summary_page = st.Page("st_pages/discussionSummary.py", title="Discussion Summary", icon=":material/sports_kabaddi:")
  reset_page = st.Page(reset, title="Try Other Paper", icon=":material/search:")
  
  page_dict = {}
  if "title" in st.session_state:
    pages = [dashboard_page, review_summary_page, inconsitency_summary_page, discussion_summary_page]
    page_dict[st.session_state.title] = pages

    pg = st.navigation({"Account": [reset_page]} | page_dict)
    st.title(st.session_state.title)
  else:
    pg = st.navigation([st.Page(get_url)])    

  pg.run()

  

if __name__ == '__main__':
  main()
