from torch.nn.functional import cosine_similarity
from metareview_generator import *
from functions import *
from similarity_evaluation import *
from sentence_transformers import util
import argparse
import itertools
import json
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# 기존 모델 리스트에 OpenAI 임베딩 모델 추가
BERT_MODELS = ["bert-base-uncased", "bert-large-uncased", "bert-base-cased", "bert-large-cased"] 
SBERT_MODELS = ["all-MiniLM-L6-v2"]
simCSE_MODELS = ["princeton-nlp/sup-simcse-bert-base-uncased"]
OPENAI_EMBEDDING_MODELS = ["text-embedding-ada-002"]  # OpenAI 임베딩 모델

# 새로운 OpenAI 임베딩 생성 함수 추가
def get_openai_embedding(model_name: str, text_chunks: list) -> FAISS:
    embeddings = OpenAIEmbeddings(model=model_name)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def main(args):
    # from args
    print(args)
    url = args.url
    assert len(url) > 0
    eval_mode = args.eval_mode
    area_chair_type = args.AC_type
    metareviewhelper = args.MVhelper

    # get ground truth metareview
    true_metareview, decision = get_true_metareivew_from_url(url)
    print("True metareview")
    print(true_metareview)

    # metareview generator    
    score, metareview = generate_metareview(url, area_chair_type, metareviewhelper)
    print("Score:", score)
    input_text = clean_text(metareview)
    print(input_text)

    # evaluation part
    if eval_mode == "BERT":
        model_name = "bert-base-cased"
        assert model_name in BERT_MODELS
        true_embed = get_bert_embedding(model_name, true_metareview)
        embed = get_bert_embedding(model_name, input_text)
        sim_score = cosine_similarity(true_embed, embed).item()

    elif eval_mode == "SBERT":
        model_name = "all-MiniLM-L6-v2"
        assert model_name in SBERT_MODELS
        true_embed = get_SBERT_embedding(model_name, true_metareview)
        embed = get_SBERT_embedding(model_name, input_text)
        sim_score = util.cos_sim(true_embed, embed).item()

    elif eval_mode == "simCSE":
        model_name = "princeton-nlp/sup-simcse-bert-base-uncased"
        assert model_name in simCSE_MODELS
        true_embed = get_simCSE_embedding(model_name, true_metareview)
        embed = get_simCSE_embedding(model_name, input_text)
        sim_score = cosine_similarity(true_embed, embed).item()

    elif eval_mode == "OpenAI":
        model_name = "text-embedding-ada-002"
        assert model_name in OPENAI_EMBEDDING_MODELS
        true_vectorstore = get_openai_embedding(model_name, [true_metareview])
        vectorstore = get_openai_embedding(model_name, [input_text])
        sim_score = true_vectorstore.similarity(vectorstore).item()

    print(sim_score)
    return score, sim_score

# 모든 조합 실행 함수 추가
def run_all_combinations(url):
    # Define possible configurations
    eval_modes = ["BERT", "SBERT", "simCSE", "OpenAI"]
    AC_types = ["inclusive", "conformist", "authoritarian", "BASELINE"]
    MV_helpers = [True, False]

    # Save results to JSON file: all at once
    dir_path = os.getcwd()
    file_path = os.path.join(dir_path, "./evaluation_results.json")
    
    # Clear contents or create an empty file
    with open(file_path, "w") as json_file:
        json_file.write("")

    results = []

    # Iterate over all combinations
    for eval_mode, AC_type, MV_helper in itertools.product(eval_modes, AC_types, MV_helpers):
        print(f"Running configuration: eval_mode={eval_mode}, AC_type={AC_type}, MV_helper={MV_helper}")

        # Prepare arguments for main function
        args = argparse.Namespace(
            url=url,
            eval_mode=eval_mode,
            AC_type=AC_type,
            MVhelper=MV_helper
        )

        # Run the main function
        score, sim_score = main(args)

        # Store the result
        result = {
            "eval_mode": eval_mode,
            "AC_type": AC_type,
            "MV_helper": MV_helper,
            "score": score,
            "similarity_score": sim_score
        }

        results.append(result)

        # Append to JSON file
        append_to_json_file(result, file_path)

    return results

def append_to_json_file(data, file_path):
    """Append a single JSON object to a file."""
    # Open file in append mode
    with open(file_path, "a") as json_file:
        json.dump(data, json_file)
        json_file.write("\n")  # Ensure each object is written on a new line

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process OpenReview arguments.")

    # Define arguments
    parser.add_argument("--url", type=str, help="OpenReview forum URL.", default="")
    parser.add_argument("--eval_mode", type=str, choices=["BERT", "SBERT", "simCSE", "OpenAI"], 
                        help="Evaluation mode to use. Options: BERT, SBERT, simCSE, OpenAI.", default="BERT")
    parser.add_argument("--AC_type", type=str, choices=["inclusive", "conformist", "authoritarian", "BASELINE"],
                         help="Area chair type. Options: inclusive, conformist, authoritarian, BASELINE.", default="BASELINE")
    parser.add_argument("--MVhelper", action="store_true",
                        help="Flag to enable metareview helper. Default is False.")
    parser.add_argument("--all_combinations", action="store_true",
                        help="Run all combinations of eval_mode, AC_type, and MV_helper.")

    args = parser.parse_args()

    args.url = "https://openreview.net/forum?id=3ULaIHxn9u7"  # manually set the URL

    if args.all_combinations:
        results = run_all_combinations(args.url)
    else:
        score, sim_score = main(args)
