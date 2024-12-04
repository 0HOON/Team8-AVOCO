

seed=3
OUTPUT_FILE="./evaluation_results$seed.json"
URL="https://openreview.net/forum?id=3ULaIHxn9u7"


python similarity_evaluation.py \
    --url $URL\
    --output_file $OUTPUT_FILE\
    --all_combinations



python visualization_results.py \
    --output_file $OUTPUT_FILE\
    --seed $seed