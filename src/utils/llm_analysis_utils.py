import pandas as pd
import json
import os
 
def process_analysis(analysis):
    """
    Process the LLM response to extract the analysis.

    Args:
        analysis (str): LLM response to process.

    Returns:
        str: Processed LLM analysis.
    """
    llm_analysis = ''
    for line in analysis.strip().split("\n"):
        if line.startswith('Process: '):
            llm_process = line.replace('Process: ', '')
            split_proc = llm_process.split(' ')
            llm_score = split_proc[-1].strip("()")
            llm_name = ' '.join(split_proc[0:-1])
        else:
            llm_analysis += line + '\n'
    
    return llm_name, llm_score, llm_analysis

def save_progress(df, response_dict, out_file):
    """
    Save DataFrame and LLM response dictionary to file.
    """
    df.to_csv(f'{out_file}.tsv', sep='\t', index=True)
    
    if os.path.exists(f'{out_file}.json'):
        with open(f'{out_file}.json', 'r') as f:
            data= json.load(f)
    else:
        data = {}
    data.update(response_dict)
    
    with open(f'{out_file}.json', 'w') as fp:
        json.dump(data, fp)
