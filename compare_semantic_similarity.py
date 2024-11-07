
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from glob import glob
import re
from scipy.stats import wilcoxon, mannwhitneyu

import os
import logging
import re

# open the file
files= glob('data/*_simVals_DF.tsv')

if not os.path.isdir('figures'):
    os.makedirs('figures', mode=0o755)
combined_df = pd.DataFrame()
order = []
for file in files:
    df = pd.read_csv(file, sep='\t')
    model_name = re.sub('^.*\/','', file).split('_processed')[0]
    print(model_name)
    order.append(model_name)
    prefix = model_name

    # if the default Name is 'system of unrelated proteins' remove it from the dataframe
    df = df[df[f'{prefix}_default Name'] != 'System of unrelated proteins']
    # print(df.shape)
    sem_sim = df.loc[:, ['GO', 'LLM_name_GO_term_sim']]
    # sem_sim = df.loc[:, ['GO', 'true_GO_term_sim_percentile']]
    sem_sim['model'] = model_name
    combined_df = pd.concat([combined_df, sem_sim])

print(combined_df.head())

fig, ax = plt.subplots(figsize=(2.5,2.5))

# Perform Mann-Whitney U tests
p_values = {}
for i in range(len(order)):
    for j in range(i+1, len(order)):
        model1 = order[i]
        model2 = order[j]
        data1 = combined_df[combined_df['model'] == model1]['LLM_name_GO_term_sim']
        data2 = combined_df[combined_df['model'] == model2]['LLM_name_GO_term_sim']
        stat, p = mannwhitneyu(data1, data2)
        p_values[(model1, model2)] = p

# Add swarm plot
sns.swarmplot(x='model', y='LLM_name_GO_term_sim', data=combined_df, ax=ax, palette=['#E07A5F', '#4579BD'], size=1.75, alpha=1)
# add median line
medians = combined_df.groupby(['model'])['LLM_name_GO_term_sim'].median()
for i in range(len(order)):
    print('model: ', order[i], 'median: ', medians[order[i]])
    ax.plot([i-0.25, i+0.25], [medians[order[i]], medians[order[i]]], lw=1, color='black')


# Determine the maximum y-value
max_y = max(combined_df['LLM_name_GO_term_sim']) + 0.03  # Adjust the 0.03 if needed

# Offset for each bar to prevent overlap
bar_offset = max_y * 0.05
i = 0
# Loop through your p-values and add bars for each significant pair
for index, ((model1, model2), p_value) in enumerate(p_values.items()):
    if p_value < 0.05:  # Only plot bars for significant differences
        bar_start = order.index(model1)  # x-coordinate for model1
        bar_end = order.index(model2)    # x-coordinate for model2

        # Adjust y-coordinate for the bar to avoid overlap
        y_coord = max_y + bar_offset + (i * bar_offset)

        # Plot the horizontal line for the significance bar
        ax.plot([bar_start, bar_end], [y_coord, y_coord], color='black', lw=0.5)

        # Add text for significance level
        significance_text = "****" if p_value < 0.0001 else "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"
        ax.text((bar_start + bar_end) / 2, y_coord - 0.01, significance_text, ha='center')
        print(f'{model1} vs {model2}: p = {p_value}')
        i += 1


ax.set_ylabel('Semantic similarity \n between LLM name and GO term name')
ax.set_xlabel('')
ax.set_xticklabels(['Mixtral\nInstruct', 'Llama2\n70b'],rotation=0, ha='center')
sns.despine()
plt.savefig('figures/compare_raw_semantic_similarity_swamp_only.png', bbox_inches='tight')



