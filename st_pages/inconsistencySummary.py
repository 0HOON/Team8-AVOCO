import streamlit as st
from utils import instructions, represent_pdf
import re

st.header("Inconsistency Summary")

# Initialize all required session state variables to ensure persistence
if 'inconsistency_summary' not in st.session_state:
    st.session_state.inconsistency_summary = None
if 'find_inconsistency' not in st.session_state:
    st.session_state.find_inconsistency = None
if 'extracted_strings' not in st.session_state:
    st.session_state.extracted_strings = None
if 'output_files' not in st.session_state:
    st.session_state.output_files = {}

if 'title' in st.session_state:
    if st.button("Summarize!"):
        with st.spinner("Processing"):
            # Save the inconsistency summary in session state
            st.session_state.inconsistency_summary = st.session_state.chain.invoke(
                instructions["inconsistency_summary"] + "\n\n summaries: " + st.session_state.full_text
            )
    
    # st.write("Full Inconsistency Summary:")

    if st.session_state.inconsistency_summary:
        # Extract each <inconsistency> block
        # st.write("Each Extracted Inconsistency Block:")
        # st.write(st.session_state.inconsistency_summary)

        reviewer_incons = re.findall(r"<Reviewer>(.*?)<\\Reviewer>", st.session_state.inconsistency_summary, re.DOTALL)
        comment_incons = re.findall(r"<Comments>(.*?)<\\Comments>", st.session_state.inconsistency_summary, re.DOTALL)
        summary_incons = re.findall(r"<inconsistency summary>(.*?)<\\inconsistency summary>", st.session_state.inconsistency_summary, re.DOTALL)

        # st.write("Reviewer Inconsistencies:")
        # st.write(reviewer_incons)
        # st.write("Comment Inconsistencies:")
        # st.write(comment_incons)
        # st.write("Summary Inconsistencies:")
        # st.write(summary_incons)

        if not summary_incons:
            st.warning("No inconsistencies found.")
        else:
            num_to_show = min(2, len(summary_incons))
            for i in range(num_to_show):
                inconsistency_key = f"find_inconsistency{i + 1}"
                
                # Process each inconsistency if not already done
                if inconsistency_key not in st.session_state:
                    st.session_state[inconsistency_key] = st.session_state.chain.invoke(
                        instructions["find_inconsistency_in_pdf"]
                        + "paper text: "
                        + st.session_state.paper_text
                        + "inconsistency_summary: "
                        + summary_incons[i]
                    )
                    st.session_state.showed_text = (
                        f"Inconsistency {i + 1}: \n {summary_incons[i]}\n\n"
                        f"Reviewers: \n"
                        + reviewer_incons[i * 2]
                        + ', '
                        + reviewer_incons[i * 2 + 1]
                        + "\n\n"
                        f"Comments: \n"
                        + comment_incons[i * 2]
                        + ', '
                        + comment_incons[i * 2 + 1]
                        + "\n\n"
                    )
                
                # Display the saved text
                st.write(st.session_state.showed_text)

                if inconsistency_key in st.session_state:
                    if f"output_file_{i}" not in st.session_state.output_files:
                        st.session_state.find_inconsistency = st.session_state.chain.invoke(
                            instructions["find_inconsistency_in_pdf"]
                            + "paper text: "
                            + st.session_state.paper_text
                            + "Inconsistency Text: "
                            + summary_incons[i]
                        )
                        st.session_state.extracted_strings = re.findall(
                            r'"(.*?)"', st.session_state.find_inconsistency
                        )
                        # represent_input= [(inconsistency_text, reviewer_id) for inconsistency_text, reviewer_id in zip(st.session_state.extracted_strings, reviewer_incons[i * 2: i * 2 + 2])]
                        st.write(st.session_state.extracted_strings)
                        output_file = represent_pdf(st.session_state.extracted_strings, reviewer_incons[i * 2: i * 2 + 2])
                        st.session_state.output_files[f"output_file_{i}"] = output_file
                    
                    # Show download button for each output file
                    with open(st.session_state.output_files[f"output_file_{i}"], "rb") as pdf_file:
                        pdf_data = pdf_file.read()
                    st.download_button(
                        label=f"Download Highlighted PDF for Inconsistency {i + 1}",
                        data=pdf_data,
                        file_name=f"inconsistency_{i + 1}_highlighted.pdf",
                        mime="application/pdf",
                    )
else:
    st.write("Please enter the OpenReview URL of the paper in the sidebar.")
