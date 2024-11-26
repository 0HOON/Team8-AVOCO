import streamlit as st
from utils import instructions

st.header(":material/sports_kabaddi: Discussion Summary")

if 'discussion_summary' in st.session_state:
  st.markdown(st.session_state.discussion_summary)
else:
  with st.spinner("Processing"):
    st.session_state.discussion_summary = st.session_state.chain.invoke(instructions["discussion_summary"] + "\n\n full text of reviews: " + st.session_state.full_text)
    st.rerun()