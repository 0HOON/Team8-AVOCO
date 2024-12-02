# Bar plot for comparison
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import argparse



EVAL_MODES = ["BERT", "SBERT", "simCSE"]

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return pd.DataFrame(data)

def save_figure_as_pdf(fig, file_path):
    """
    Save a matplotlib figure as a PDF file.

    Parameters:
    - fig: The matplotlib figure object to save.
    - file_path: The path to save the PDF file, including the file name and extension.
    """
    fig.savefig(file_path, format='pdf', bbox_inches='tight')



def plot_bar_plots(df, seed):
    
    
    for eval_mode in EVAL_MODES:
        subset = df[df["eval_mode"] == eval_mode]
        ac_types = subset["AC_type"].unique()
        
        # Prepare data for bar plot
        scores_true = subset[subset["MV_helper"] == True].set_index("AC_type")["similarity_score"].reindex(ac_types, fill_value=0)
        scores_false = subset[subset["MV_helper"] == False].set_index("AC_type")["similarity_score"].reindex(ac_types, fill_value=0)
        
        x = np.arange(len(ac_types))
        width = 0.35  # Bar width

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

        # MV_helper = True
        bars_true = axes[0].bar(x, scores_true, color='blue', alpha=0.7)
        axes[0].set_title("MV_helper = True")
        axes[0].set_xlabel("AC_type")
        axes[0].set_ylabel("Similarity Score")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(ac_types, rotation=45)
        # Add annotations
        for bar in bars_true:
            height = bar.get_height()
            axes[0].annotate(f"{height:.2f}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # Offset text by 3 points
                            textcoords="offset points",
                            ha="center", va="bottom")

        # MV_helper = False
        bars_false = axes[1].bar(x, scores_false, color='orange', alpha=0.7)
        axes[1].set_title("MV_helper = False")
        axes[1].set_xlabel("AC_type")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(ac_types, rotation=45)
        # Add annotations
        for bar in bars_false:
            height = bar.get_height()
            axes[1].annotate(f"{height:.2f}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # Offset text by 3 points
                            textcoords="offset points",
                            ha="center", va="bottom")



        # Adjust layout and save as PDF
        fig.suptitle(f"Similarity Score by AC_type for {eval_mode}")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig_path = f"./{eval_mode}_plots_{seed}.pdf"
        save_figure_as_pdf(fig=fig, file_path=fig_path)



def main(args):
    
    df = read_json_file(file_path=args.output_file)

    plot_bar_plots(df, args.seed)

if __name__ == "__main__":



    parser = argparse.ArgumentParser()
    
    # Define arguments
    parser.add_argument("--output_file", type=str, help="json evaluation file.", default ="./evaluation_results2.json")
    parser.add_argument("--seed", type=int, help="seed.", required=True)
    args = parser.parse_args()

    main(args)