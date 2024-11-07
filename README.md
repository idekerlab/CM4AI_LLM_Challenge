# CM4AI_LLM_Challenge

## set up environment 

```
conda create -n llm_challenge python=3.11.5
conda activate llm_challenge
pip install -r requirements.txt
```

## Toy example:

Located in ./data/toy_example_w_contaminated.csv

| Column                  | Description                                              |
|-------------------------|----------------------------------------------------------|
| GO                      | Gene Ontology Term ID                                    |
| Genes                   | Genes assigned to this GO term                           |
| Gene Count              | Number of genes                                          |
| Term_Description        | GO term name                                             |
| 50perc_contaminated_Genes | Contaminated gene set with 50% real and 50% random       |
| 100perc_contaminated_Genes | Complete random gene set                                |

## Task 1 run the pipeline in bash script 
```
mkdir -p logs

# adjust URL to location below to where ollama service is hosted
export LOCAL_MODEL_HOST=http://localhost:11434/api/chat

bash ./run_LLM_annotation_pipeline.sh
```
## Task 2 compare semantic similarity with GO names 

```
bash ./excute_semantic_similarity_calculation.sh
```
[Visualize the semantic similarity result](./Task3.Compare_semantic_similarity.ipynb)

## Task 3 compare confidence scores 
Check out notebook for visualizing [Task2](./Task2.Compare_confidence_scores.ipynb) 



## Optional Task 4 run on gene sets of interest from VNN interpretation

**example run command line**
```
# run in the command line  
input_file='data/GO_term_analysis/toy_example.csv' # replace with your file
config='./jsonFiles/llama2_run_VNN_sample.json' #check this example config and make a new config for your own task 
set_index='GO' # update this to your index of choice
gene_column='Genes' # update to the gene column
gene_sep=' ' # update based on what separator you have
start=0
end=11 # update the number for the length of the file  
out_file = 'data/LLM_processed_VNN_genesets_llama2'
python query_llm_for_analysis.py --config $config \
            --initialize \ # initialize the analysis column with None
            --input $input_file \
            --input_sep  ','\ # update to '\t' if tsv
            --set_index $set_index \
            --gene_column $gene_column\
            --gene_sep ' ' \
            --start $start \
            --end $end \
            --output_file $out_file
```
