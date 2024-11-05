#!/bin/bash 
source activate llm_challenge

input_file='data/GO_term_analysis/toy_example.csv'
set_index='GO'
gene_column='Genes'
gene_sep=' '
start=0
end=11   
# Define model options
model_options=('mixtral_instruct' 'llama2_70b')
for model in "${model_options[@]}"; do 
    output_file="data/${model}_processed_toy_example"
    config_file="jsonFiles/toyexample_${model}.yaml"
    python query_llm_for_analysis.py --config $config_file --input_file $input_file --initialize --set_index $set_index --gene_column $gene_column --gene_sep $gene_sep --run_contaminated --start $start --end $end --model $model --output_file $output_file