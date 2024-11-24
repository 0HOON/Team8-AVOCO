import re
import functions
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver

functions.load_dotenv()

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
    return "## Your Biography ## \n\n"

def add_information_needed(metareviewhelper: bool, texts: tuple, url: str) -> str:
    """
    Adds information needed for writing the meta-review to the prompt.
    :param metareviewhelper: Boolean flag to include meta-review assistance.
    :param texts: Tuple containing paper content and reviews.
    :param url: URL to fetch paper content and reviews.
    :return: Information string.
    """
    reviews, paper_content = texts

    prompt = "Here are three pieces of information needed to write a meta-review: \n"
    prompt += "1. Paper contents \n 2. Reviews \n"

    if metareviewhelper:
        prompt += "3. Information provided by the meta-review assistant service: \n \t 3.1 Review summary \n \t 3.2 Reviewers' inconsistency summary \n \t 3.3 Discussion summary\n"

    prompt += "## Paper contents ##\n"
    prompt += paper_content + "\n"

    prompt += "## Reviews ##\n"
    prompt += reviews + "\n"

    if metareviewhelper:
        chain = functions.prepare_chain(url)[0]
        prompt += "## Information provided by the meta-review assistant service ##\n"
        prompt += chain.invoke(functions.instructions["review_summary"] + "\n\n reviews: " + reviews) + "\n"
        prompt += chain.invoke(functions.instructions["inconsistency_summary"] + "\n\n reviews: " + reviews) + "\n"
        prompt += chain.invoke(functions.instructions["discussion_summary"] + "\n\n reviews: " + reviews) + "\n"

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

# Example usage
if __name__ == "__main__":
    url = "https://openreview.net/forum?id=3ULaIHxn9u7"
    area_chair_type = "inclusive"
    metareviewhelper = True
    score, metareview = generate_metareview(url, area_chair_type, metareviewhelper)
    print("Score:", score)
    print("Metareview:", metareview)
