import streamlit as st
from utils import instructions, tokens, parse_text

st.header(":material/sports_kabaddi: Discussion Summary")

def display_text():
   for i in range(len(st.session_state.discussion_summary['discussion'])):
    with st.expander(f"**{st.session_state.discussion_summary['discussion'][i]}**"):
      cols = st.columns((1, 1), gap='medium')
      col_i = 0
      for token in tokens["discussion_summary"]:
        if token in ["discussion", "writer"]:
          continue
        with cols[col_i]:
          with st.container(border=True):
            st.markdown(f"#### {token.capitalize()}")
            for sentence in st.session_state.discussion_summary[token][i].strip().split(". "):
              st.markdown(f"* {sentence}")
        col_i = (col_i + 1)%2

if 'discussion_summary' in st.session_state:
  display_text()
else:
  text = ""
  for i, review in enumerate(st.session_state.root.replies):
    with st.spinner(f"Processing Discussion # {i+1}"):
      if review.writer != "Authors":
        text += st.session_state.chain.invoke(
          instructions["discussion_summary"] 
          + "\n\n Review and replies: " 
          + review.get_text(0, recursive=True)
        )
    break
  st.session_state.discussion_summary = parse_text(text, tokens["discussion_summary"])
  st.rerun()