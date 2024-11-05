#!/bin/bash -l

source activate llm_challenge

# Loop over files and calculate semantic similarity
for file in data/*processed_toy_example.tsv; do
# Run the Python script for each file
    model=$(basename "$file" .tsv | awk -F'_' '{print $(NF-1)"_"$NF}')
    model_name_col='$(model)_default Name'
    GO_name_col='Term_Description'
    echo "Processing $file"

    python calculate_semantic_similarity.py --inputFile $file --nameCol1 $GO_name_col --nameCol2 $model_name_col

    python 
    date
done