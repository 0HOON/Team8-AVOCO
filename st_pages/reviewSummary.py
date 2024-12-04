import streamlit as st
from utils import instructions, tokens, parse_text

st.header(":material/emoji_people: Review Summary")

if 'review_summary' in st.session_state:
  for i in range(len(st.session_state.review_summary['review'])):
    with st.expander(f"**{st.session_state.review_summary['reviewer'][i]}**", ):
      st.markdown(f"### Summary")
      st.markdown(f"{st.session_state.review_summary['review summary'][i]}")
      st.markdown(f"### Strengths")
      st.markdown(f"{st.session_state.review_summary['strength'][i]}")
      st.markdown(f"### Weaknesses")
      st.markdown(f"{st.session_state.review_summary['weakness'][i]}")
      st.markdown(f"### Keywords")
      st.markdown(f"{st.session_state.review_summary['keywords'][i]}")
      st.markdown(f"### Important Pages")
      st.markdown(f"{st.session_state.review_summary['important pages'][i]}")
else:
  with st.spinner("Processing"):
    st.session_state.text = st.session_state.chain.invoke(instructions["review_summary"] + "\n\n full text of reviews: " + st.session_state.full_text)
    st.session_state.review_summary = parse_text(st.session_state.text, tokens["review_summary"])
    st.rerun()