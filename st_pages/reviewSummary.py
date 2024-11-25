import streamlit as st
from utils import instructions

st.header("Review Summary")
if 'title' in st.session_state:
  if st.button("Summarize!"):
    with st.spinner("Processing"):
      st.session_state.review_summary = st.session_state.chain.invoke(instructions["review_summary"] + "\n\n summaries: " + st.session_state.full_text)
  if 'review_summary' in st.session_state:
      st.write(st.session_state.review_summary)
else:
  st.write("Please enter the OpenReview URL of the paper in the sidebar")