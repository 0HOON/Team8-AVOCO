import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel,BertModel, BertTokenizer
from torch.nn.functional import cosine_similarity
from scipy.spatial.distance import cosine
from metareview_generator import *
from functions import *
import re
from sentence_transformers import SentenceTransformer, util
import argparse
import itertools
import json
import os



BERT_MODELS = ["bert-base-uncased","bert-large-uncased", "bert-base-cased", "bert-large-cased" ] 
SBERT_MODELS = ["all-MiniLM-L6-v2", "multi-qa-mpnet-base-dot-v1", "all-distilroberta-v1"]
simCSE_MODELS = ["princeton-nlp/sup-simcse-bert-base-uncased"]

def get_bert_embedding(model_name  :str, text : str) -> torch.Tensor:
    '''
    get mean pooling embedding
    average all vectors of all tokens
    '''
    tokenizer = BertTokenizer.from_pretrained(model_name)

    model = BertModel.from_pretrained(model_name)
    model.eval()


        # Encode text
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    
    # Compute token embeddings
    with torch.no_grad():
        output = model(**encoded_input)
     
    # Mean pooling - take attention mask into account for correct averaging
    embeddings = output.last_hidden_state
    attention_mask = encoded_input['attention_mask']
    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    mean_pooled_embeddings = sum_embeddings / sum_mask

    return mean_pooled_embeddings


def get_bert_embedding_cls(model_name: str, text: str) -> torch.Tensor:
    '''
    get [cls] token embedding
    '''

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()

    # Tokenize text
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        output = model(**encoded_input)

    # Extract [CLS] token embedding (first token)
    cls_embedding = output.last_hidden_state[:, 0, :]  # Shape: [batch_size, hidden_size]

    return cls_embedding




def get_SBERT_embedding(model_name :str, sentence : str):
    model = SentenceTransformer(model_name)
    return model.encode(sentence)

def get_simCSE_embedding(model_name  :str, text : str) ->torch.Tensor: 


    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize input texts
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

# Get the embeddings with attention mask consideration
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.last_hidden_state  # Hidden states from the last layer
        attention_mask = inputs['attention_mask']  # Attention mask for valid tokens

        # Expand attention mask dimensions
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()

        # Perform mean pooling
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1)
        mean_pooled_embeddings = sum_embeddings / torch.clamp(sum_mask, min=1e-9)  # Avoid division by zero

    return mean_pooled_embeddings



def get_simCSE_embedding_cls(model_name: str, text: str) -> torch.Tensor:

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize input texts
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    # Get the embeddings
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        
        # Extract the [CLS] token embeddings
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # Shape: [batch_size, hidden_size]

    return cls_embeddings


def clean_text(input_text : str) -> str: 
    # Remove 'Score: 7' and leading/trailing whitespace
    cleaned_text = re.sub(r"Score: \d+\n\n", "", input_text).strip()
    return cleaned_text






def get_true_metareivew_from_url(url, venue_id="ICLR.cc/2023/Conference") -> str:
    client = openreview.Client()
    paper_id = url.split('=')[-1]
    paper_info = client.get_note(paper_id)
    reviews = client.get_notes(replyto=paper_id, details='replies')
    program_chair = [note for note in reviews if "Program_Chairs" in note.signatures[0]][0]

    assert program_chair is not None
    paper_decision = program_chair.content["decision"]
    metareview = program_chair.content['metareview:_summary,_strengths_and_weaknesses']
    output = format_sections_with_eos(metareview)

    return output, paper_decision

def format_sections_with_eos(input_text) -> str:
    # Define patterns for each section
    summary_pattern = r"(?<=Summary:\n)(.*?)(?=\n\n|Strengths:|Weakness:)"
    strengths_pattern = r"(?<=Strengths:\n)(.*?)(?=\n\n|Weakness:)"
    weakness_pattern = r"(?<=Weakness:\n)(.*)"

    # Extract sections using regex
    summary = re.search(summary_pattern, input_text, re.DOTALL)
    strengths = re.search(strengths_pattern, input_text, re.DOTALL)
    weakness = re.search(weakness_pattern, input_text, re.DOTALL)

    # Get group content or set as None if not found
    summary = summary.group(1).strip() if summary else "None"
    strengths = strengths.group(1).strip() if strengths else "None"
    weakness = weakness.group(1).strip() if weakness else "None"

    # Format the sections with <EOS>
    formatted_text = f"Summary: {summary}<EOS>\nStrengths: {strengths}<EOS>\nWeakness: {weakness}<EOS>".strip()
    return formatted_text


def main(args):
    

    #from args
    # url = "https://openreview.net/forum?id=3ULaIHxn9u7"
    # eval_mode = "BERT" #BERT, SBERT, simCSE
    # area_chair_type = "inclusive" # 'inclusive', 'conformist', 'authoritarian', 'BASELINE'
    # metareviewhelper = True # False

    print(args)
    url = args.url
    assert len(url)>0
    eval_mode = args.eval_mode
    area_chair_type = args.AC_type
    metareviewhelper = args.MVhelper
    

    #get ground truth metareview
    true_metareview, decision = get_true_metareivew_from_url(url)
    print("True metareview")
    print(true_metareview)


    #metareview generator    
    score, metareview = generate_metareview(url, area_chair_type, metareviewhelper)
    print("Score:", score)
    # print("Metareview:", metareview)
    input_text = clean_text(metareview)
    print(input_text)

    #evaluation part
    if eval_mode == "BERT":
        
        model_name = "bert-base-cased"
        assert model_name in BERT_MODELS
        true_embed = get_bert_embedding_cls(model_name, true_metareview)
        embed = get_bert_embedding_cls(model_name, input_text)
        sim_score = cosine_similarity(true_embed, embed).item()
        

    elif eval_mode =="SBERT":
        model_name = "multi-qa-mpnet-base-dot-v1" # ["all-MiniLM-L6-v2", "multi-qa-mpnet-base-dot-v1", "all-distilroberta-v1"]
        assert model_name in SBERT_MODELS
        true_embed = get_SBERT_embedding(model_name, true_metareview)
        embed = get_SBERT_embedding(model_name, input_text)
        sim_score = util.cos_sim(true_embed, embed).item()

    else: #simCSE
        model_name = "princeton-nlp/sup-simcse-bert-base-uncased"
        assert model_name in simCSE_MODELS
        true_embed = get_simCSE_embedding_cls(model_name, true_metareview)
        embed = get_simCSE_embedding_cls(model_name, input_text)
        sim_score = cosine_similarity(true_embed, embed).item()

    print(sim_score)

    return score, sim_score , model_name
    



#save json
def append_to_json_file(data, file_path):
    """Append a single JSON object to a file."""
    # Open file in append mode
    with open(file_path, "a") as json_file:
        json.dump(data, json_file)
        json_file.write("\n")  # Ensure each object is written on a new line


def run_all_combinations(url, output_file):
    # Define possible configurations
    eval_modes = ["BERT", "SBERT", "simCSE"] #["SBERT"]
    AC_types = ["inclusive", "conformist", "authoritarian", "BASELINE"]
    MV_helpers = [True, False]


    # Save results to JSON file : all at once
    dir_path = os.getcwd()
    file_path = os.path.join(dir_path ,output_file)
    
    with open(file_path, "w") as json_file:
        json_file.write("")  # Clear contents or create an empty file



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
        score, sim_score,model_name = main(args)

        # Store the result
        result = {
            "eval_mode": eval_mode,
            "model_name": model_name,
            "AC_type": AC_type,
            "MV_helper": MV_helper,
            "score": score,
            "similarity_score": sim_score
        }

        results.append(result)

        append_to_json_file(result, file_path)

    return results


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Process OpenReview arguments.")
    
    # Define arguments
    parser.add_argument("--url", type=str, help="OpenReview forum URL.", default ="https://openreview.net/forum?id=3ULaIHxn9u7")
    parser.add_argument("--eval_mode", type=str, choices=["BERT", "SBERT", "simCSE"], 
                        help="Evaluation mode to use. Options: BERT, SBERT, simCSE.", default = "BERT")
    parser.add_argument("--AC_type", type=str, choices=["inclusive", "conformist", "authoritarian", "BASELINE"],
                         help="Area chair type. Options: inclusive, conformist, authoritarian, BASELINE.", default = "BASELINE")
    parser.add_argument("--MVhelper", action="store_true",
                        help="Flag to enable metareview helper. Default is False.")
    parser.add_argument("--all_combinations", action="store_true",
                        help="Run all combinations of eval_mode, AC_type, and MV_helper.")
    parser.add_argument("--output_file", type=str, help="json output_file", default ="./evaluation_results2.json")
    
    args = parser.parse_args()


    if args.all_combinations:
        output_file =args.output_file
        results = run_all_combinations(args.url, output_file=output_file)
    else:
        score, sim_score,_= main(args)