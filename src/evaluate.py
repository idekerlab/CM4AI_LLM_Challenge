import logging
import os
import pickle
import re

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import wandb

from utils.llm_analysis_utils import process_analysis, save_progress
from utils.prompt_factory import make_user_prompt_with_score
from utils.semanticSimFunctions import getNameSimilarities_no_repeat
from utils.server_model_query import server_model_chat

os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)

## Load the sapbert model and tokenizer
SapBERT_tokenizer = AutoTokenizer.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
SapBERT_model = AutoModel.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')


def get_logger(filename):
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler(filename)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def llm_annotate(df, base_filename, config):
    analysis_dict  = {}
    
    LOG_FILE = f"logs/{base_filename}_{config['ind_start']}_{config['ind_end']}.log"
    OUT_FILE = f"results/{base_filename}_processed"
    logger = get_logger(f'logs/{base_filename}.log')

    if "customized_prompt" in config and config["customized_prompt"] is not None:
        # make sure the file exist 
        if os.path.isfile(config['customized_prompt']):
            with open(config['customized_prompt'], 'r') as f: # replace with your actual customized prompt file
                customized_prompt = f.read()
                assert len(customized_prompt) > 1, "Customized prompt is empty"
                wandb.save(config['customized_prompt'])
        else:
            print("Customized prompt file does not exist")
            customized_prompt = None
    else:
        customized_prompt = None

    i = 0 #used for track progress and saving the file
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        #only process None rows 
        if pd.notna(row[f'{column_prefix} Analysis']):
            continue
        
        gene_data = row[config["gene_column"]]
        # if gene_data is not a string, then skip
        if type(gene_data) != str:
            
            logger.warning(f'Gene set {idx} is not a string, skipping')
            continue
        genes = gene_data.split(config["gene_sep"])
        
        if len(genes) >1000:
            logger.warning(f'Gene set {idx} is too big, skipping')
            continue

        try:
            prompt = make_user_prompt_with_score(genes, customized_prompt=customized_prompt)
            # print(prompt)
            print("Querying LLM: ", config["model"])
            analysis, error_message= server_model_chat(
                config["context"],
                prompt,
                config["model"],
                config["temperature"],
                config["max_tokens"],
                LOG_FILE,
                config["seed"]
            )
            
            if analysis:
                logger.info(analysis)
                llm_name, llm_score, llm_analysis = process_analysis(analysis)
                # clean up the score and return float
                logger.info('llm_name: ' + llm_name)
                logger.info('llm_score: ' + llm_score)
                try:
                    llm_score_value =  float(re.sub("[^0-9.-]", "", llm_score))
                except ValueError:
                    llm_score_value = llm_score
                
                df.loc[idx, f'{column_prefix} Name'] = llm_name
                df.loc[idx, f'{column_prefix} Analysis'] = llm_analysis
                df.loc[idx, f'{column_prefix} Score'] = llm_score_value
                analysis_dict[f'{idx}_{column_prefix}'] = analysis
                # Log success with fingerprint
                logger.info(f'Success for {idx} {column_prefix}.')
            else:
                if error_message:
                    logger.error(f'Error for query gene set {idx}: {error_message}')
                else:
                    logger.error(f'Error for query gene set {idx}: No analysis returned')
                    
        except Exception as e:
            logger.error(f'Error for {idx}: {e}')
            continue

        i += 1
        if i % 10 == 0:
            # bin scores into no score, low, medium, high confidence
            bins = [-np.inf, 0, 0.81, 0.86, np.inf] # 0 is no score (name not assigned), between 0 to 0.8 is low confidence, between 0.82 to 0.86 is medium confidence, above 0.87 is high confidence
            labels = ['Name not assigned', 'Low Confidence', 'Medium Confidence', 'High Confidence']  # Define the corresponding labels
            
            df[f'{column_prefix} Score bins'] = pd.cut(df[f'{column_prefix} Score'], bins=bins, labels=labels)
            save_progress(df, analysis_dict, OUT_FILE)
            # df.to_csv(f'{out_file}.tsv', sep='\t', index=True)
            print(f"Saved progress for {i} genesets")
    # save the final file
    # bin scores into no score, low, medium, high confidence
    bins = [-np.inf, 0, 0.81, 0.86, np.inf] # 0 is no score (name not assigned), between 0 to 0.8 is low confidence, between 0.82 to 0.86 is medium confidence, above 0.87 is high confidence
    labels = ['Name not assigned', 'Low Confidence', 'Medium Confidence', 'High Confidence']  # Define the corresponding labels
    
    df.loc[:,f'{column_prefix} Score bins'] = pd.cut(df[f'{column_prefix} Score'], bins=bins, labels=labels)
    save_progress(df, analysis_dict, OUT_FILE)

    return f"{OUT_FILE}.tsv"


def save_emb_dict(emb_dict, file_name):
    with open(file_name, 'wb') as handle:  
        pickle.dump(emb_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def calc_sim(inputFile, nameCol_LLM, nameCol_GO):
    reduced_LLM_genes_DF = pd.read_csv(inputFile, sep = "\t") 
    reduced_LLM_genes_DF[nameCol_GO] = reduced_LLM_genes_DF[nameCol_GO].replace(np.nan, 'NaN')
    
    ## initialize the dataframe with dummy values
    new_DF = reduced_LLM_genes_DF.copy()
    new_DF['LLM_name_GO_term_sim'] = None
    
    # skip rows with LLM Name as 'system of unrelated proteins' ignore cases
    
    filtered_DF = reduced_LLM_genes_DF[reduced_LLM_genes_DF[nameCol_LLM].str.lower() != 'system of unrelated proteins'].reset_index(drop = True)
    skipped_rows = reduced_LLM_genes_DF[reduced_LLM_genes_DF[nameCol_LLM].str.lower() == 'system of unrelated proteins'].reset_index(drop = True)
    
    llm_name_embedding_dict = {}
    go_term_embedding_dict = {}
    names_DF, llm_emb_dict, go_emb_dict = getNameSimilarities_no_repeat(filtered_DF, nameCol_LLM, nameCol_GO, 
    SapBERT_tokenizer, SapBERT_model,llm_name_embedding_dict,
        go_term_embedding_dict, "cosine_similarity")
    
    # add back the rows with LLM Name as 'system of unrelated proteins'
    names_DF = pd.concat([names_DF, skipped_rows]).reset_index(drop = True)

    save_emb_dict(llm_emb_dict, inputFile.replace(".tsv", "_llm_emb_dict.pkl"))
    save_emb_dict(go_emb_dict, inputFile.replace(".tsv", "_go_emb_dict.pkl"))
    names_DF.to_csv(inputFile.replace(".tsv", "_simVals_DF.tsv"), sep = "\t", index = False)

    return names_DF


##################
##################
##################
if __name__ == "__main__":
    config = {
        "context": "You are an efficient and insightful assistant to a molecular biologist",
        "temperature": 0.0,
        "model": "mixtral:instruct",
        "max_tokens": 1000,
        "ind_start": 0,
        "ind_end": 11,
        "gene_sep": ' ',
        "input_sep": ',',
        "gene_column": "Genes",
        "set_index": "GO",
        "dataset_name": "toy_example_w_cont",
        "input_file": "data/toy_example_w_contaminated.csv",
        "seed": 42,
        "run_contaminated": True,
        "customized_prompt": "data/custom_prompt_2.txt"
    }

    with wandb.init(
            # set the wandb project where this run will be logged
            project="llm-challenge-wls",
            entity="b2ai-cm4ai",
            # track hyperparameters and run metadata
            config=config
        ) as run:

        # TODO: simplify this logic
        model_name_fix = config["model"]
        if '-' in config["model"]:
            model_name_fix = '_'.join(config["model"].split('-')[:2])
        else:
            model_name_fix = config["model"].replace(':', '_')
        column_prefix = model_name_fix + '_default' #start from default gene set

        base_filename = f"{config['dataset_name']}_{model_name_fix}"
        
        if config["input_sep"] == '\\t':
            config["input_sep"] = '\t'

        raw_df = pd.read_csv(config["input_file"], sep=config["input_sep"], index_col=config["set_index"])
        
        # Only process the specified range of genes
        df = raw_df.iloc[config["ind_start"]:config["ind_end"]].copy(deep=True)

        df.loc[:,f'{column_prefix} Name'] = None
        df.loc[:,f'{column_prefix} Analysis'] = None
        df.loc[:,f'{column_prefix} Score'] = -np.inf

        print(df[f'{column_prefix} Analysis'].isna().sum())
        output_file = llm_annotate(df, base_filename, config)  ## run with the real set 

        # if run_contaminated is true, then run the pipeline for contaminated gene sets
        if config["run_contaminated"]:
            ## run the pipeline for contaiminated gene sets 
            contaminated_columns = [col for col in df.columns if col.endswith('contaminated_Genes')]
            # print(contaminated_columns)
            for col in contaminated_columns:
                config["gene_column"] = col ## Note need to change the gene_column to the contaminated column
                contam_prefix = '_'.join(col.split('_')[0:2])
                
                column_prefix = model_name_fix + '_' + contam_prefix
                print(column_prefix)
                
                df.loc[:, f'{column_prefix} Name'] = None
                df.loc[:, f'{column_prefix} Analysis'] = None
                df.loc[:, f'{column_prefix} Score'] = -np.inf
                
                print(df[f'{column_prefix} Analysis'].isna().sum())
                llm_annotate(df, base_filename, config)

        sim_results = calc_sim(os.path.join(output_file), f"{model_name_fix}_default Name", "Term_Description")
        table = wandb.Table(dataframe=sim_results)
        for col_name in sim_results.columns.tolist():
            if col_name.lower().endswith('score'):
                wandb.run.summary[col_name] = sim_results['col_name'].mean()
        run.log({"sem_sim_results": table})
