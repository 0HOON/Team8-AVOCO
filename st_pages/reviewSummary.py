import streamlit as st
import re
from utils import instructions, tokens, parse_text


def parse_opinion(text, tokens):
  opinions = text.split("<opinion>")
  parsed_opinions = []
  for opinion in opinions:
    parsed_data = {}
    if opinion == "":
      continue
    # Regular expression to extract content between the token tags
    for token in tokens:
      if token == "opinion":
        parsed_data["title"] = opinion.split("</opinion>")[0]
        continue
      pattern = f"<{token}>(.*?)</{token}>"
      matches = re.findall(pattern, opinion, re.DOTALL)  # DOTALL to include newlines
      parsed_data[token] = matches
    parsed_opinions.append(parsed_data)
  return parsed_opinions


def display_opinion():
  for i, opinion in enumerate(st.session_state.opinion_summary):
    title = ""
    title = f"{opinion['title']}"
    with st.expander(title):
      st.markdown("#### Summary")
      st.markdown(f"- {opinion['opinion summary'][0]}")
      st.markdown("#### Comments")
      for reviewer, comment in zip(opinion["reviewer"], opinion["comments"]):
        with st.container(border=True):
          st.markdown(f"**{reviewer}**: *{comment.strip()}*")
      st.markdown(f"#### Relevant Pages")
      st.markdown(f"- {opinion['important pages'][0]}")

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

## Page   
st.header(":material/emoji_people: Strength & Weakness Summary")

if 'review_summary' in st.session_state:
  display_text()
  st.header(":material/lightbulb: Opinion Summary")
  display_opinion()

else:
  review_out = ""
  opinion_out = ""
  review_text = ""
  for i, review in enumerate(st.session_state.root.replies):
    with st.spinner(f"Analyzing Reviews... ({i+1}/{len(st.session_state.root.replies)})"):
      if review.writer != "Authors":
        review_out += st.session_state.chain.invoke(
          instructions["review_summary"] 
          + "\n\n Full text of review: " 
          + review.get_text(0, recursive=False)
        )
        review_text += review.get_text(0, recursive=False)

  with st.spinner(f"Analyzing Opinions..."):
    opinion_out = st.session_state.chain.invoke(
      instructions["opinion_summary"] 
      + "\n\n Full text of review: " 
      + review_text
    )

  st.session_state.review_summary = parse_text(review_out, tokens["review_summary"])
  st.session_state.opinion_summary = parse_opinion(opinion_out, tokens["opinion_summary"])
  st.rerun()