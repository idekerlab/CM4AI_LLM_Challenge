import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import re


def melt_score_df(score_df, id_vars=['GO', 'Gene_Count']):
    columns_score = [col for col in score_df.columns if col.endswith('Score')]
    filtered_score_df = score_df[id_vars + columns_score]
    long_scores = filtered_score_df.melt(id_vars=id_vars, var_name='Score Type', value_name='Score')

    # Map the original score type names to the desired ones
    def map_score_type(score_type):
        if 'default' in score_type:
            return 'Real'
        elif '50perc_contaminated' in score_type:
            return '50/50 Mix'
        elif '100perc_contaminated' in score_type:
            return 'Random'
        else:
            return score_type

    # Apply the mapping
    long_scores['Score Type'] = long_scores['Score Type'].apply(map_score_type)

    # Remove unwanted symbols and return float
    long_scores.Score = long_scores.Score.apply(lambda x: float(re.sub("[^0-9.-]", "", x)) if isinstance(x, str) else x)

    return long_scores

# combine the results
result_files = glob('./data/*_processed_toy_example.tsv')
print(len(result_files))
combine_df = pd.DataFrame()
model_order = []
for file in result_files:
    df = pd.read_csv(file, sep='\t')
    long_scores = melt_score_df(df)

    model_name = re.sub('^.*\/', '', file).split('_processed')[0]
    model_order.append(model_name)
    print(model_name)

    # map the model name to my own choice of names
    long_scores['Model'] = model_name
    combine_df = pd.concat([combine_df, long_scores])

combine_df.head()

# plot the confidence score bins
# Defining the custom order for Score Types and Models
score_type_order = ['Real', '50/50 Mix', 'Random']
score_bin_order = ['Name not assigned', 'Low Confidence', 'Medium Confidence', 'High Confidence']
color_palette = ['#A5AA99', '#e39cc5', '#af549e', '#6c2167']

# Creating a categorical type for ordering
combine_df['Score Type'] = pd.Categorical(combine_df['Score Type'], categories=score_type_order, ordered=True)
combine_df['Model'] = pd.Categorical(combine_df['Model'], categories=model_order, ordered=True)
combine_df['Score Bin'] = pd.Categorical(combine_df['Score Bin'], categories=score_bin_order, ordered=True)

# Grouping and stacking data
stacked_bins_data = combine_df.groupby(['Model', 'Score Type', 'Score Bin']).size().unstack().fillna(0)

# Modify the layout to include an additional subplot
fig, axes = plt.subplots(1, len(model_order) + 1, figsize=(7, 3), sharey=True)

# Plotting for each model (existing code)
for i, model in enumerate(model_order):
    model_data = stacked_bins_data.loc[model]
    model_data.plot(kind='bar', stacked=True, ax=axes[i], width=0.8, legend=False, color=color_palette)
    axes[i].set_xlabel(model)
    axes[i].set_xticklabels(['Real', '50/50 Mix', 'Random'], rotation=45)
    if i == 0:
        axes[i].set_ylabel('Counts')
    else:
        axes[i].yaxis.set_visible(False)

# Adjusting the layout for the new plot
for i, ax in enumerate(axes):
    if i != 0:
        sns.despine(ax=ax, left=True)
    else:
        sns.despine(ax=ax)

plt.tight_layout()

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(handles, labels, title='Score Bin', loc='upper left', bbox_to_anchor=(1, 1), fontsize=6)
plt.tight_layout(h_pad=0.01)
# Saving the figure
plt.savefig('./figures/confidence_score_compare.png', bbox_inches='tight')