import json
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from metareview_generator import generate_metareview_all_ac_type
from similarity_evaluation import get_openai_embedding, get_true_metareview_from_url

load_dotenv()

DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MODEL = "gpt-4o"
OPENAI_MODELS = ["gpt-4o", "text-embedding-ada-002"]

def calculate_similarity(true_metareview: str, generated_metareview: str, model_name: str) -> float:
    """
    Calculate the similarity score between the true metareview and a generated meta-review.
    """
    true_embed = get_openai_embedding(model_name, true_metareview)
    generated_embed = get_openai_embedding(model_name, generated_metareview)
    sim_score = cosine_similarity(true_embed.unsqueeze(0), generated_embed.unsqueeze(0)).item()
    return sim_score

def generate_similarity_json(results_all, model_name: str):
    """
    Generate JSON data for similarity scores between true metareview and generated meta-reviews.
    """
    json_results = []
    for result in results_all:
        url, ac_type, score, generated_metareview, metareviewhelper = result

        # Get true metareview
        true_metareview, paper_decision = get_true_metareview_from_url(url)
        sim_score = calculate_similarity(true_metareview, generated_metareview, model_name)

        result_data = {
            "URL": url,
            "model_name": model_name,
            "AC_type": ac_type,
            "MV_helper": "with" if metareviewhelper else "without",
            "score": score,
            "similarity_score": sim_score,
            "paper_decision": paper_decision
        }
        json_results.append(result_data)

    return json_results

def plot_similarity_scores(results_with_helper, results_without_helper, ac_types):
    """
    Plot the average similarity scores for each AC type, comparing with and without meta-review helpers.
    """
    # Prepare data for plotting
    def prepare_scores(results, ac_types):
        scores = {ac: [] for ac in ac_types}
        for result in results:
            scores[result["AC_type"].upper()].append(result["similarity_score"])
        avg_scores = [np.mean(scores[ac]) for ac in ac_types]
        return avg_scores, scores

    avg_scores_with, scores_with = prepare_scores(results_with_helper, ac_types)
    avg_scores_without, scores_without = prepare_scores(results_without_helper, ac_types)

    # Calculate improvement for each AC type when metareview helper is used
    improvements = []
    for i, ac_type in enumerate(ac_types):
        if avg_scores_without[i] != 0:
            improvement = (avg_scores_with[i] - avg_scores_without[i]) / avg_scores_without[i] * 100
        else:
            improvement = 0
        improvements.append(improvement)

    # Plotting the two bar charts side by side
    x = np.arange(len(ac_types))
    width = 0.35
    y_min = min(min(avg_scores_with), min(avg_scores_without))
    y_max = max(max(avg_scores_with), max(avg_scores_without))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot for metareview helper = True
    ax1.bar(x, avg_scores_with, capsize=5, width=width, label='With Helper', color='b')
    ax1.set_xlabel('Area Chair Types')
    ax1.set_ylabel('Average Similarity Scores')
    ax1.set_title(f'Average Similarity Scores with Meta-Review Helper\nStd Dev: {np.std(avg_scores_with):.5f}')
    ax1.set_xticks(x)
    ax1.set_xticklabels(ac_types)
    ax1.set_ylim(y_min, y_max)
    ax1.legend()

    # Plot for metareview helper = False
    ax2.bar(x, avg_scores_without, capsize=5, width=width, label='Without Helper', color='r')
    ax2.set_xlabel('Area Chair Types')
    ax2.set_ylabel('Average Similarity Scores')
    ax2.set_title(f'Average Similarity Scores without Meta-Review Helper\nStd Dev: {np.std(avg_scores_without):.5f}')
    ax2.set_xticks(x)
    ax2.set_xticklabels(ac_types)
    ax2.set_ylim(y_min, y_max)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("comparison_subplots/similarity_scores_comparison_side_by_side.png")
    plt.show()

    # Plot improvements as a separate table
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    table_data = [[ac_types[i], f'{improvements[i]:.2f}%'] for i in range(len(ac_types))]
    table = ax.table(cellText=table_data, colLabels=['Area Chair Type', 'Improvement (%)'], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    plt.tight_layout()
    plt.savefig("comparison_subplots/improvement_table.png")
    plt.show()

def plot_normalized_similarity_scores(results_with_helper, results_without_helper, ac_types):
    """
    Plot the normalized average similarity scores for each AC type, comparing with and without meta-review helpers.
    """
    # Prepare data for plotting
    def prepare_scores(results, ac_types):
        scores = {ac: [] for ac in ac_types}
        for result in results:
            scores[result["AC_type"].upper()].append(result["similarity_score"])
        avg_scores = [np.mean(scores[ac]) for ac in ac_types]
        return avg_scores, scores

    avg_scores_with, scores_with = prepare_scores(results_with_helper, ac_types)
    avg_scores_without, scores_without = prepare_scores(results_without_helper, ac_types)

    # Combine all similarity scores for normalization
    all_scores = [score for ac_scores in scores_with.values() for score in ac_scores] + [score for ac_scores in scores_without.values() for score in ac_scores]
    mean_all = np.mean(all_scores)
    std_all = np.std(all_scores)

    # Normalize the scores by subtracting the mean and dividing by the standard deviation (using all scores)
    normalized_with = (avg_scores_with - mean_all) / std_all
    normalized_without = (avg_scores_without - mean_all) / std_all

    # Calculate improvement for each AC type when metareview helper is used (normalized)
    improvements = []
    for i, ac_type in enumerate(ac_types):
        if normalized_without[i] != 0:
            improvement = (normalized_with[i] - normalized_without[i]) / abs(normalized_without[i]) * 100
        else:
            improvement = 0
        improvements.append(improvement)

    # Plotting the two bar charts side by side
    x = np.arange(len(ac_types))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Set common y-axis limits
    y_min = min(normalized_with.min(), normalized_without.min()) * 1.1 if min(normalized_with.min(), normalized_without.min()) < 0 else min(normalized_with.min(), normalized_without.min()) * 0.9
    y_max = max(normalized_with.max(), normalized_without.max()) * 1.1

    # Plot for metareview helper = True (normalized)
    ax1.bar(x, normalized_with, capsize=5, width=width, label='With Helper (Normalized)', color='b')
    ax1.set_xlabel('Area Chair Types')
    ax1.set_ylabel('Normalized Similarity Scores')
    ax1.set_title(f'Normalized Similarity Scores with Meta-Review Helper Std Dev: {np.std(normalized_with):.5f}')
    ax1.set_xticks(x)
    ax1.set_xticklabels(ac_types)
    ax1.set_ylim(y_min, y_max)
    ax1.legend()

    # Plot for metareview helper = False (normalized)
    ax2.bar(x, normalized_without, capsize=5, width=width, label='Without Helper (Normalized)', color='r')
    ax2.set_xlabel('Area Chair Types')
    ax2.set_ylabel('Normalized Similarity Scores')
    ax2.set_title(f'Normalized Similarity Scores without Meta-Review Helper Std Dev: {np.std(normalized_without):.5f}')
    ax2.set_xticks(x)
    ax2.set_xticklabels(ac_types)
    ax2.set_ylim(y_min, y_max)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("comparison_subplots/normalized_similarity_scores_comparison_side_by_side.png")
    plt.show()

    # Plot normalized improvements as a separate table
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    table_data = [[ac_types[i], f'{improvements[i]:.2f}%'] for i in range(len(ac_types))]
    table = ax.table(cellText=table_data, colLabels=['Area Chair Type', 'Improvement (Normalized %)'], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)


    plt.tight_layout()
    plt.savefig("comparison_subplots/normalized_improvement_table.png")
    plt.show()

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

    # Calculate true similarity scores
    model_name = "text-embedding-ada-002"
    true_similarity_results_with_helper = generate_similarity_json(results_true_all, model_name)
    true_similarity_results_without_helper = generate_similarity_json(results_false_all, model_name)

    # Save to JSON
    with open('eval_results/true_metareview_similarity_results_with_helper.json', 'w') as json_file:
        json.dump(true_similarity_results_with_helper, json_file, indent=4)

    with open('eval_results/true_metareview_similarity_results_without_helper.json', 'w') as json_file:
        json.dump(true_similarity_results_without_helper, json_file, indent=4)

    # Plot similarity scores
    plot_similarity_scores(true_similarity_results_with_helper, true_similarity_results_without_helper, area_chair_types)

    # Plot normalized similarity scores
    plot_normalized_similarity_scores(true_similarity_results_with_helper, true_similarity_results_without_helper, area_chair_types)

if __name__ == "__main__":
    main()
