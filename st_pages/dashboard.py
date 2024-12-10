import streamlit as st
import pandas as pd
import altair as alt

st.header(":material/browse_activity: Score Distribution")

def extract_reviews_and_scores(replies):
    # Fields to extract from content
    score_fields = [
        "correctness",
        "technical_novelty_and_significance",
        "empirical_novelty_and_significance",
        "recommendation",
        "confidence"
    ]

    # Initialize data dictionary
    extracted_data = {
        "reviewers": [],
        "correctness": [],
        "technical_novelty_and_significance": [],
        "empirical_novelty_and_significance": [],
        "recommendation": [],
        "confidence": []
    }

    # Safeguard: Ensure all elements are valid objects
    for i, review in enumerate(replies):
        if hasattr(review, "writer") and hasattr(review, "content"):
            writer = review.writer
            reviewer_id = writer.split('/')[-1].split('_')[-1]  # Extract reviewer ID after "_"
            if len(reviewer_id) == 4:
                extracted_data["reviewers"].append(reviewer_id)

            # Extract scores from content
            for field in score_fields:
                value = review.content.get(field)  # Default to 'N/A' if field is missing
                if value:  # Only append valid data
                    score = value.split(':')[0].strip()  # Extract numeric score
                    extracted_data[field].append(score)
        else:
            print(f"Skipping invalid reply at index {i}: {review}")  # Debug invalid entries

    return extracted_data

# Example Usage
if hasattr(st.session_state, "root") and st.session_state.root.replies:
    reviews_data = extract_reviews_and_scores(st.session_state.root.replies)
    


df = pd.DataFrame(reviews_data)

# Top Row
col1, col2 = st.columns(2, gap="medium")
with col1:
    st.markdown("### Technical Novelty and Significance")
    source = df[["reviewers", "technical_novelty_and_significance"]]
    source.rename(columns={"reviewers": "Reviewer", "technical_novelty_and_significance": "Score"}, inplace=True)

    st.altair_chart(
        alt.Chart(source)
        .mark_bar()
        .encode(
            x="Reviewer:N",
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 5])),
        )
        .properties(
            width=300,
            height=300
        )
        .configure_axisY(
            tickMinStep=1
        )
    )

with col2:
    st.markdown("### Empirical Novelty and Significance")
    source = df[["reviewers", "empirical_novelty_and_significance"]]
    source.rename(columns={"reviewers": "Reviewer", "empirical_novelty_and_significance": "Score"}, inplace=True)

    st.altair_chart(
        alt.Chart(source)
        .mark_bar()
        .encode(
            x="Reviewer:N",
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 5])),
        )
        .properties(
            width=300,
            height=300
        )
        .configure_axisY(
            tickMinStep=1
        )
    )

# Bottom Row
col3, col4, col5 = st.columns(3, gap="medium")

with col3:
    st.markdown("### Correctness")
    source = df[["reviewers", "correctness"]]
    source.rename(columns={"reviewers": "Reviewer", "correctness": "Score"}, inplace=True)

    st.altair_chart(
        alt.Chart(source)
        .mark_bar()
        .encode(
            x="Reviewer:N",
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 5])),
        )
        .properties(
            width=300,
            height=300
        )
        .configure_axisY(
            tickMinStep=1
        )
    )

with col4:
    st.markdown("### Recommendation")
    source = df[["reviewers", "recommendation"]]
    source.rename(columns={"reviewers": "Reviewer", "recommendation": "Score"}, inplace=True)

    st.altair_chart(
        alt.Chart(source)
        .mark_bar()
        .encode(
            x="Reviewer:N",
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 10])),
        )
        .properties(
            width=300,
            height=300
        )
        .configure_axisY(
            tickMinStep=1
        )
    )

with col5:
    st.markdown("### Confidence")
    source = df[["reviewers", "confidence"]]
    source.rename(columns={"reviewers": "Reviewer", "confidence": "Score"}, inplace=True)

    st.altair_chart(
        alt.Chart(source)
        .mark_bar()
        .encode(
            x="Reviewer:N",
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 5])),
        )
        .properties(
            width=300,
            height=300
        )
        .configure_axisY(
            tickMinStep=1
        )
    )
    