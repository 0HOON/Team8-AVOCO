import streamlit as st
from utils import instructions, represent_pdf
import re

st.header("Inconsistency Summary")
if 'title' in st.session_state:
  if st.button("Summarize!"):
    with st.spinner("Processing"):
      st.session_state.inconsistency_summary = st.session_state.chain.invoke(instructions["inconsistency_summary"] + "\n\n summaries: " + st.session_state.full_text)
  if 'inconsistency_summary' in st.session_state:
      st.write(st.session_state.inconsistency_summary)
      if st.button("Find Inconsistency in PDF!"):
        with st.spinner("Processing"):
          st.session_state.find_inconsistency = st.session_state.chain.invoke(instructions["find_inconsistency_in_pdf"] + "paper text: " + st.session_state.paper_text+ "inconsistency_summary: "+ st.session_state.inconsistency_summary)
          
      if 'find_inconsistency' in st.session_state:
          st.write(st.session_state.find_inconsistency)
          extracted_strings=re.findall(r'"(.*?)"',st.session_state.find_inconsistency)
          output_file=represent_pdf(extracted_strings)
          with open(output_file, "rb") as pdf_file:
             pdf_data=pdf_file.read()
          st.download_button(
              label="Download Highlighted PDF",
              data=pdf_data,
              file_name=output_file,
              mime="application/pdf"
          )

          #st.write(st.session_state.find_inconsistency)
else:
  st.write("Please enter the OpenReview URL of the paper in the sidebar")