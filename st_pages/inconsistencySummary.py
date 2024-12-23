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
    if st.session_state.inconsistency_summary is None:
        with st.spinner("Analyzing Inconsistencies..."):
            # Save the inconsistency summary in session state
            st.session_state.inconsistency_summary = st.session_state.chain.invoke(
                instructions["inconsistency_summary"] + "\n\n reviews: " + st.session_state.full_text
            )
        st.rerun()
    else:
        # Extract inconsistencies
        reviewer_incons = re.findall(r"<Reviewer>(.*?)<\\Reviewer>", st.session_state.inconsistency_summary, re.DOTALL)
        comment_incons = re.findall(r"<Comments>(.*?)<\\Comments>", st.session_state.inconsistency_summary, re.DOTALL)
        summary_incons = re.findall(r"<inconsistency summary>(.*?)<\\inconsistency summary>", st.session_state.inconsistency_summary, re.DOTALL)

        if not summary_incons:
            st.warning("No inconsistencies found.")
        else:
            with st.spinner():
                for i, summary in enumerate(summary_incons):
                    inconsistency_key = f"find_inconsistency{i + 1}"

                    # Process each inconsistency if not already done
                    if inconsistency_key not in st.session_state:
                        st.session_state[inconsistency_key] = st.session_state.chain.invoke(
                            instructions["find_inconsistency_in_pdf"]
                            + "paper text: "
                            + st.session_state.paper_text
                            + "inconsistency_summary: "
                            + summary
                        )


                    # Group reviewers and comments dynamically
                    paired_data = list(zip(reviewer_incons[i::len(summary_incons)], comment_incons[i::len(summary_incons)]))

                    if f"output_file_{i}" not in st.session_state.output_files:
                        st.session_state.find_inconsistency = st.session_state.chain.invoke(
                            instructions["find_inconsistency_in_pdf"]
                            + "paper text: "
                            + st.session_state.paper_text
                            + "Inconsistency Text: "
                            + summary
                        )
                        st.session_state.extracted_strings = re.findall(
                            r'"(.*?)"', st.session_state.find_inconsistency
                        )
                        output_file = represent_pdf(st.session_state.extracted_strings, [pair[0] for pair in paired_data])
                        st.session_state.output_files[f"output_file_{i}"] = output_file

                    # Display inconsistencies
                    with st.expander(f"Inconsistency {i + 1}"):
                        st.markdown("#### Summary")
                        st.markdown(f"{summary}")
                        st.markdown("#### Comments")
                        for reviewer, comment in paired_data:
                            with st.container(border=True):
                                st.markdown(f"**{reviewer}**: {comment}")
                        # Show download button for each output file
                        with open(st.session_state.output_files[f"output_file_{i}"], "rb") as pdf_file:
                            pdf_data = pdf_file.read()
                        st.download_button(
                            label=f":material/download: Download Highlighted PDF for Inconsistency {i + 1}",
                            data=pdf_data,
                            file_name=f"inconsistency_{i + 1}_highlighted.pdf",
                            mime="application/pdf",
                        )
else:
    st.write("Please enter the OpenReview URL of the paper in the sidebar.")
