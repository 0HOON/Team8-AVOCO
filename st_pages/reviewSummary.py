import streamlit as st
from utils import instructions

st.header(":material/emoji_people: Review Summary")

if 'review_summary' in st.session_state:
  st.markdown(st.session_state.review_summary)
else:
  with st.spinner("Processing"):
    st.session_state.review_summary = st.session_state.chain.invoke(instructions["review_summary"] + "\n\n summaries: " + st.session_state.full_text)
    st.rerun()