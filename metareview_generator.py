import re
import functions
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MODEL = "gpt-4o"

def generate_metareview(url: str, area_chair_type: str, metareviewhelper: bool) -> tuple:
    """
    Generates a meta-review based on the paper content and reviewer evaluations.
    :param url: URL to fetch paper content and reviews.
    :param area_chair_type: Type of the area chair ('inclusive', 'conformist', 'authoritarian', 'BASELINE').
    :param metareviewhelper: Boolean flag to include meta-review assistance.
    :return: Tuple of score (float) and meta-review (string).
    """
    texts = functions.get_texts_from_url(url)
    prompt = create_global_prompt()
    prompt += add_area_chair_description(area_chair_type)
    prompt += add_information_needed(metareviewhelper, texts, url)
    prompt += add_metareview_guideline()
    prompt += add_rubrics()
    prompt += add_output_format()

    response = generate_metareview_from_chain(prompt)
    score = extract_score(response)
    return score, response

def create_global_prompt() -> str:
    """
    Creates the global prompt explaining the situation.
    :return: Global prompt string.
    """
    prompt = "You are a very knowledgeable and experienced area chair in a top-tier machine learning conference. "
    prompt += "You evaluate the reviews provided by reviewers and write metareviews. Later, you will decide which paper gets accepted or rejected based on your metareviews. "
    return prompt

def add_area_chair_description(area_chair_type: str) -> str:
    """
    Adds the area chair description to the prompt based on the type.
    :param area_chair_type: Type of the area chair ('inclusive', 'conformist', 'authoritarian', 'BASELINE').
    :return: Area chair description string.
    """
    desc_inclusive_ac = ("You are an inclusive area chair. You tend to hear from all reviewers' opinions and combine "
                         "them with your own judgments to make the final decision.\n\n")

    desc_conformist_ac = ("You are a conformist area chair who perfunctorily handles area chair duties. You mostly follow "
                          "the reviewers' suggestions to write your metareview, score the paper, and decide whether "
                          "to accept a paper.\n\n")

    desc_authoritarian_ac = ("You are an authoritarian area chair. You tend to read the paper on your own, follow your "
                             "own judgment, and mostly ignore the reviewers' opinions.\n\n")

    if area_chair_type == "inclusive":
        return f"## Your Biography ## \n{desc_inclusive_ac}"
    elif area_chair_type == "conformist":
        return f"## Your Biography ## \n{desc_conformist_ac}"
    elif area_chair_type == "authoritarian":
        return f"## Your Biography ## \n{desc_authoritarian_ac}"
    elif area_chair_type == "ALL":
        return f"""## Biography description ## \n 
        INCLUSIVE : {desc_inclusive_ac} \n
        CONFORMIST : {desc_conformist_ac} \n
        AUTHORITARIAN : {desc_authoritarian_ac} \n
        BASELINE : \n
        \n
        Please actively refer to the above Biography descriptions when scoring the paper and writing your meta-reviews. It is perfectly acceptable for the scores and evaluations to vary based on these descriptions, as they are intended to guide a more nuanced and informed assessment.\n
        """
    return "## Your Biography ## \n\n"

def add_information_needed(metareviewhelper: bool, texts: tuple, url: str) -> str:
    """
    Adds information needed for writing the meta-review to the prompt.
    :param metareviewhelper: Boolean flag to include meta-review assistance.
    :param texts: Tuple containing reviews, paper content, and PDF content.
    :param url: URL to fetch additional content if needed.
    :return: Information string.
    """
    reviews, paper_content, pdf = texts

    prompt = "Here are three pieces of information needed to write a meta-review:\n"
    prompt += "1. Paper contents\n2. Reviews\n"

    if metareviewhelper:
        prompt += ("3. Information provided by the meta-review assistant service:\n"
                   "\t3.1 Review summary\n"
                   "\t3.2 Reviewers' inconsistency summary\n"
                   "\t3.3 Discussion summary\n")
    else:
        prompt += "3. (No additional information provided by the meta-review assistant service)\n"

    prompt += "## Paper contents ##\n"
    prompt += paper_content + "\n"

    prompt += "## Reviews ##\n"
    prompt += reviews + "\n"

    if metareviewhelper:
        chain = functions.prepare_chain(url)
        prompt += "## Information provided by the meta-review assistant service ##\n"
        review_summary = chain.invoke(functions.instructions["review_summary"] + "\n\nreviews: " + reviews)
        inconsistency_summary = chain.invoke(functions.instructions["inconsistency_summary"] + "\n\nreviews: " + reviews)
        discussion_summary = chain.invoke(functions.instructions["discussion_summary"] + "\n\nreviews: " + reviews)

        prompt += review_summary + "\n"
        prompt += inconsistency_summary + "\n"
        prompt += discussion_summary + "\n"

        # Add instruction to actively refer to this content
        prompt += ("\nPlease actively refer to the information provided above by the meta-review assistant service, regardless of the biography description, when scoring the paper and writing your meta-review.\n")

    return prompt

def prepare_simple_chain():
    """
    Prepares a simple LangChain prompt chain without using vector embeddings or Streamlit components.
    :return: Prepared LangChain prompt chain.
    """
    llm = ChatOpenAI(model_name="gpt-4o", temperature=DEFAULT_TEMPERATURE)
    parser = StrOutputParser()

    prompt_chain = (
        llm
        | parser
    )

    return prompt_chain

def add_metareview_guideline() -> str:
    """
    Adds the metareview guideline to the prompt.
    :return: Metareview guideline string.
    """
    guideline = "Here are the guidelines for writing meta-reviews and the rubric to follow when assigning scores\n"
    guideline += """ ##Metareview Guideline##
As an AC, we trust you to make an informed recommendation based on sufficient knowledge and justified analysis of the paper and to clearly and thoroughly convey this recommendation and reasoning behind it to the authors. To this end, you have full freedom in writing your meta-reviews, although we list below a few items that have been found useful by authors when they were presented with meta-reviews. Aim to write a meta-review of at least 60 words.

1. A concise description of the submission’s main content (scientific claims and findings) based on your own reading and reviewers’ characterization, including the paper’s strengths and weaknesses. Ideally this description should contain both what is discussed in the submission and what is missing from the submission.
2. A concise summary of discussion. Unlike other conferences in which there is only a single round of back-and-forth between reviewers and authors, ICLR distinguishes itself by providing many weeks of discussion. These weeks of discussion and meetings not only serve the purpose of decision making but also to contribute scientifically to improve the submission. We thus encourage the AC to summarize the discussion in the meta-review. In particular, it is advised that the AC lists the points that were raised by the reviewers, how each of these points was addressed by the authors and whether you as the AC found each point worth consideration in decision making.
3. Your recommendation and justification. The meta-review should end with a clear indication of your recommendation. Your recommendation must be justified based on the content and discussion of the submission (i.e., the points you described above).

"""
    return guideline

def add_rubrics() -> str:
    """
    Adds the rubric for scoring to the prompt.
    :return: Rubric string.
    """
    SCORE_CALCULATION = {
        10: "This study is among the top 2% of all papers. It is one of the most thorough I have seen. It changed my thinking on this topic. I would fight for it to be accepted",
        8: "This study is among the top 10% of all papers. It provides sufficient support for all of its claims/arguments. Some extra experiments are needed, but not essential. The method is highly original and generalizable to various fields. It deepens the understanding of some phenomenons or lowers the barriers to an existing research direction",
        6: "This study provides sufficient support for its major claims/arguments, some minor points may need extra support or details. The method is moderately original and generalizable to various relevant fields. The work it describes is not particularly interesting and/or novel, so it will not be a big loss if people don’t see it in this conference",
        5: "Some of the main claims/arguments are not sufficiently supported, there are major technical/methodological problems. The proposed method is somewhat original and generalizable to various relevant fields. I am leaning towards rejection, but I can be persuaded if my co-reviewers think otherwise",
        3: "This paper makes marginal contributions",
        1: "This study is not yet sufficiently thorough to warrant publication or is not relevant to the conference"
    }

    RUBRICS = (f"* 10: {SCORE_CALCULATION[10]};\n"
               f"* 8: {SCORE_CALCULATION[8]};\n"
               f"* 6: {SCORE_CALCULATION[6]};\n"
               f"* 5: {SCORE_CALCULATION[5]};\n"
               f"* 3: {SCORE_CALCULATION[3]};\n"
               f"* 1: {SCORE_CALCULATION[1]}. ")

    return f"## Rubrics for Overall Rating\n\n{RUBRICS}\n\n"

def add_output_format() -> str:
    """
    Adds the output format specification to the prompt.
    :return: Output format string.
    """
    output_format = "Here is the format you MUST follow when responding.\n"
    output_format += "##Output format##\n"
    output_format += "Write a metareview using the following format:\n\n"
    output_format += "``\n"
    output_format += (
        "Score: ... # Provide a score for the paper in the range from 1 to 10. Do not write any reasons. Intermediary integer scores such as 9, 7, 4, and 2 are allowed. Fractions such as "
        "6.5 is allowed.\n\n")
    output_format += ("Summary: ... <EOS> \n # Provide a summary of the paper based on the paper contents (if provided), reviewers' reviews and discussions (if provided), authors' rebuttal, and your own expertise. ")
    output_format += ("Strengths: ... <EOS> \n # Provide the paper's strengths")
    output_format += ("Weaknesses: ... <EOS> \n # Provide the paper's Weaknesses")
    output_format += "``\n\n"

    return output_format

def generate_metareview_from_chain(prompt: str) -> str:
    """
    Generates the meta-review using LangChain's prepared prompt chain.
    :param prompt: Prompt to generate the meta-review.
    :return: Generated meta-review string.
    """
    chain = prepare_simple_chain()
    response = chain.invoke(prompt)
    return response

def extract_score(response: str) -> float:
    """
    Extracts the score from the meta-review response.
    :param response: The generated meta-review string.
    :return: Extracted score as a float.
    """
    score_match = re.search(r"Score:\s*([\d\.]+)", response)
    if score_match:
        return float(score_match.group(1))
    return 0.0

def add_combined_output_format(area_chair_types: list) -> str:
    """
    Adds a combined output format specification to the prompt for all area chair types.
    :param area_chair_types: List of area chair types.
    :return: Combined output format string.
    """
    output_format = "## Output format for all area chair types ##\n"
    output_format += "Write a metareview tailored to each area chair type, taking their biography descriptions into account. Please use the following format:\n\n"
    for ac_type in area_chair_types:
        output_format += f"### {ac_type.upper()} ###\n"
        output_format += "``\n"
        output_format += (
            "Score: ... # Provide a score for the paper in the range from 1 to 10. Do not write any reasons. Intermediary integer scores such as 9, 7, 4, and 2 are allowed. Fractions such as "
            "6.5 are allowed.\n\n")
        output_format += ("Summary: ... <EOS> \n # Provide a summary of the paper based on the paper contents (if provided), reviewers' reviews and discussions (if provided), authors' rebuttal, and your own expertise. ")
        output_format += ("Strengths: ... <EOS> \n # Provide the paper's strengths")
        output_format += ("Weaknesses: ... <EOS> \n # Provide the paper's weaknesses")
        output_format += "``\n\n"

    return output_format

def extract_multiple_scores_and_reviews(response: str, area_chair_types: list) -> list:
    import re
    result = []
    for ac_type in area_chair_types:
        # Build a regex pattern to match the section for this type
        pattern = r'###\s*' + re.escape(ac_type) + r'\s*###(.*?)(?=###|$)'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            section = match.group(1).strip()
            # Extract the score
            score_match = re.search(r'Score:\s*(\d+)', section)
            if score_match:
                score = float(score_match.group(1))
            else:
                score = None
            # Add to result
            result.append((ac_type, score, section))
    return result

def generate_metareview_all_ac_type(url: str, metareviewhelper: bool) -> list:
    """
    Generates meta-reviews for all types of area chairs.
    :param url: URL to fetch paper content and reviews.
    :param metareviewhelper: Boolean flag to include meta-review assistance.
    :return: List of tuples containing scores and meta-reviews for all area chair types.
    """
    area_chair_types = ['INCLUSIVE', 'CONFORMIST', 'AUTHORITARIAN', 'BASELINE']
    texts = functions.get_texts_from_url(url)
    prompt = create_global_prompt()
    prompt += add_information_needed(metareviewhelper, texts, url)
    prompt += add_metareview_guideline()
    prompt += add_rubrics()
    prompt += add_area_chair_description("ALL")
    prompt += add_combined_output_format(area_chair_types)
    response = generate_metareview_from_chain(prompt)
    print(response)
    results = extract_multiple_scores_and_reviews(response, area_chair_types)

    return results

# Example code that compares the score variance across different area chair types with metareviewhelper True and False
def calculate_ac_type_variances(results):
    ac_type_scores = {}
    for ac_type, score, _ in results:
        if ac_type not in ac_type_scores:
            ac_type_scores[ac_type] = []
        if score is not None:
            ac_type_scores[ac_type].append(score)
    
    # Calculate mean scores and variances
    mean_scores = {ac_type: np.mean(scores) for ac_type, scores in ac_type_scores.items()}
    variances = {ac_type: np.var(scores) for ac_type, scores in ac_type_scores.items()}
    
    return mean_scores, variances

def plot_ac_type_scores(results_true, results_false, area_chair_types, output_file='ac_type_scores.pdf'):
    # Calculate variances and mean scores for both metareviewhelper True and False
    mean_scores_true, variances_true = calculate_ac_type_variances(results_true)
    mean_scores_false, variances_false = calculate_ac_type_variances(results_false)

    print("variances_true : ", variances_true)
    print("\n")
    print("variances_false : ", variances_false)


    # Set up the plots
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: Mean scores for metareviewhelper=True
    axs[0].bar(area_chair_types, [mean_scores_true[ac_type] for ac_type in area_chair_types])
    axs[0].set_title('Average Scores with MetaReviewHelper=True')
    axs[0].set_ylabel('Average Score')
    axs[0].set_xlabel('Area Chair Type')
    axs[0].set_ylim(0, 10)
    axs[0].set_xticklabels(area_chair_types, rotation=45)
    
    # Right plot: Mean scores for metareviewhelper=False
    axs[1].bar(area_chair_types, [mean_scores_false[ac_type] for ac_type in area_chair_types])
    axs[1].set_title('Average Scores with MetaReviewHelper=False')
    axs[1].set_ylabel('Average Score')
    axs[1].set_xlabel('Area Chair Type')
    axs[1].set_ylim(0, 10)
    axs[1].set_xticklabels(area_chair_types, rotation=45)

    # Display the plots
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

# Example usage of the plotting function
def main():
    urls = [
        "https://openreview.net/forum?id=N3kGYG3ZcTi",
        "https://openreview.net/forum?id=0z_cXcu1N6o",
        "https://openreview.net/forum?id=9Zx6tTcX0SE",
        "https://openreview.net/forum?id=3uDXZZLBAwd",
        "https://openreview.net/forum?id=Xj9V-stmIcO",
        "https://openreview.net/forum?id=4F1gvduDeL",
        "https://openreview.net/forum?id=6BHlZgyPOZY",
        "https://openreview.net/forum?id=dKkMnCWfVmm",
        "https://openreview.net/forum?id=7YfHla7IxBJ",
        "https://openreview.net/forum?id=KRLUvxh8uaX"
    ]

    area_chair_types = ['INCLUSIVE', 'CONFORMIST', 'AUTHORITARIAN', 'BASELINE']

    results_true_all = []
    results_false_all = []
    for url in urls:
        results_true = generate_metareview_all_ac_type(url, metareviewhelper=True)
        results_false = generate_metareview_all_ac_type(url, metareviewhelper=False)
        
        results_true_all.extend(results_true)
        results_false_all.extend(results_false)

    # Plotting the scores
    plot_ac_type_scores(results_true_all, results_false_all, area_chair_types, output_file='ac_type_scores.pdf')

if __name__ == "__main__":
    main()
