#!/bin/bash 

input_file='data/toy_example_w_contaminated.csv'
set_index='GO'
gene_column='Genes'
gene_sep=' '
input_sep=','
start=0
end=11   
# Define model options
model_options=('mixtral_instruct' 'llama2_13b' 'llama2_70b' 'llama3.1_70b' 'mixtral_8x22b' 'gemma2_27b')
for model in "${model_options[@]}"; do 
    output_file="data/${model}_processed_toy_example"
    config_file="jsonFiles/toyexample_${model}.json"
    python query_llm_for_analysis.py --config $config_file --input $input_file --input_sep "$input_sep" --initialize --set_index $set_index --gene_column $gene_column --gene_sep "$gene_sep" --run_contaminated --start $start --end $end --output_file $output_file
done
