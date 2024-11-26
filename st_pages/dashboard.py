import streamlit as st
import altair as alt
import pandas as pd

col1, col2 = st.columns((1, 1), gap='medium')
with col1:
  st.markdown("## Before")
  source = pd.DataFrame({
    'a': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
    'b': [28, 55, 43, 91, 81, 53, 19, 87, 52]
  })

  st.altair_chart(
    alt.Chart(source)
    .mark_bar(cornerRadius=15)
    .encode(
      x='a',
      y='b',
    )
    .properties(
      width=400,
    )
  )
with col2:
  st.markdown("## After")
  source = pd.DataFrame({
    'a': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
    'b': [28, 55, 43, 91, 81, 53, 19, 87, 52]
  })

  st.altair_chart(
    alt.Chart(source)
    .mark_bar(cornerRadius=15)
    .encode(
      x='a',
      y='b',
    )
    .properties(
      width=400,
    )
  )

col1, col2, col3 = st.columns((1, 1, 1))
with col1:
  st.metric(label="test1", value=100, delta=10)

with col2:
  st.metric(label="test2", value=100, delta=10)

with col3:
  st.metric(label="test3", value=100, delta=10)
