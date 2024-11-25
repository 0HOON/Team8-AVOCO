import streamlit as st
from utils import instructions

st.header(":material/connect_without_contact: Inconsistency Summary")

if 'inconsistency_summary' in st.session_state:
  st.write(st.session_state.inconsistency_summary)
else:
  with st.spinner("Processing"):
    st.session_state.inconsistency_summary = st.session_state.chain.invoke(instructions["inconsistency_summary"] + "\n\n summaries: " + st.session_state.full_text)
    st.rerun()