import streamlit as st
from dotenv import load_dotenv

from utils import prepare_chain, get_paper_list, instructions
load_dotenv(override=True)

## TODO review 전부 가져오기
## TODO 결과 복사 버튼 만들기

def select_paper():
  st.header("Meta Review Helper :books:")
  option = st.selectbox(
    "Which paper needs to be summarized?",
    options=st.session_state.paper_list,
    index=None,
    format_func=lambda x: x.content["title"],
    placeholder="Select paper...",
  )
  if option:
    st.session_state.chain = prepare_chain(option)
    option = None
    st.rerun()

def reset():
  tmp = st.session_state.paper_list
  st.session_state.clear()
  st.session_state.paper_list = tmp
  st.rerun()

def main():
  # load env settings
  # session_state list
  # - url: OpenReview URL input
  # - title: title of the paper
  # - chain: langchain object
  # - review/inconsistency/discussion_summary: generated summary text
  # - paper_list: all paper list
  st.set_page_config(page_title="Meta Review Helper", page_icon=":books:", layout="wide")
  with open("./.streamlit/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

  if "paper_list" not in st.session_state:
    with st.spinner("Initializing..."):
      st.session_state.paper_list = get_paper_list()
  
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

    pg = st.navigation({"Settings": [reset_page]} | page_dict)
    st.title(st.session_state.title)
  else:
    pg = st.navigation([st.Page(select_paper)])

  pg.run()

  

if __name__ == '__main__':
  main()
