import streamlit as st
from utils import instructions

st.header("Discussion Summary")
if 'title' in st.session_state:
  if st.button("Summarize!"):
    with st.spinner("Processing"):
      st.session_state.discussion_summary = st.session_state.chain.invoke(instructions["discussion_summary"] + "\n\n summaries: " + st.session_state.full_text)
  if 'discussion_summary' in st.session_state:
      st.write(st.session_state.discussion_summary)
else:
  st.write("Please enter the OpenReview URL of the paper in the sidebar")