import streamlit as st
from utils import instructions, tokens, parse_text


  
st.header(":material/emoji_people: Review Summary")
def display_text():
   for i in range(len(st.session_state.review_summary['review'])):
    with st.expander(f"{st.session_state.review_summary['reviewer'][i]}", ):
      cols = st.columns((1, 1), gap='medium')
      for token in tokens["review_summary"]:
        if token in ["review", "reviewer"]:
          continue        
        col_i = token != "review summary"

        with cols[col_i]:
          with st.container(border=True):
            st.markdown(f"#### {token.capitalize()}")
            for sentence in st.session_state.review_summary[token][i].strip().split(". "):
              st.markdown(f"* {sentence}")
          col_i = (col_i+1)%2
  
if 'review_summary' in st.session_state:
  display_text()

else:
  text = ""
  for i, review in enumerate(st.session_state.root.replies):
    with st.spinner(f"Processing Review {i+1}"):
      if review.writer != "Authors":
        text += st.session_state.chain.invoke(
          instructions["review_summary"] 
          + "\n\n Full text of review: " 
          + review.get_text(0, recursive=False)
        )

  st.session_state.review_summary = parse_text(text, tokens["review_summary"])
  st.rerun()