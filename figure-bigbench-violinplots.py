import numpy as np
import pandas as pd
import json
# don't let matplotlib use xwindows
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pylab import savefig
import seaborn as sns
sns.set_style("whitegrid")
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=False, default='Pythia-6.9B', type=str)
parser.add_argument('--similarity_type', required=False, default='max', type=str)
args = parser.parse_args()

if args.model == 'Pythia-6.9B':
    model = 'pythia-6.9b'
    model_name = 'Pythia'
elif args.model == 'T5':
    model = 't5-v1_1-xl'
    model_name = 'T5'
else:
    print(f'Model not found: {args.model}')
    exit()

if args.similarity_type == 'max':
    cosine_sim = 'max'
    cosine_sim_field = 'Max'
    cosine_sim_label = 'Max'
    ymin = 0.82
elif args.similarity_type == 'mean':
    cosine_sim = 'mean_top'
    cosine_sim_field = 'Mean'
    cosine_sim_label = 'Mean'
    ymin = 0.78
else:
    print(f'Similarity type not found: {args.similarity_type}')
    exit()

df = pd.read_csv(f'./raw-results/cosine-similarities/{cosine_sim}-cosine_bigbench_the-pile.csv')
with open(f'./raw-results/bigbench-lite-mul_{model}_dataset_to_index_to_correct.json', 'r') as f:
    dataset_to_index_to_correct = json.load(f)

datasets = set(dataset_to_index_to_correct.keys())
datasets = set(['hindu_knowledge', 'bbq_lite_json', 'code_line_description', 'conceptual_combinations', 'emoji_movie', 'formal_fallacies_syllogisms_negation', 'known_unknowns', 'language_identification',
                    'logic_grid_puzzle', 'logical_deduction', 'novel_concepts', 'play_dialog_same_or_different',
                    'strange_stories', 'strategyqa', 'symbol_interpretation', 'vitaminc_fact_verification', 'winowhy'])

def update_dataset_display_name(row):
    if row['Dataset'] == 'formal_fallacies_syllogisms_negation':
        return 'formal_fallacies'
    return row['Dataset']

def add_correct_fn(row):
    if row['Dataset'] not in datasets:
        return 'n/a'
    if dataset_to_index_to_correct[row['Dataset']][str(int(row['Idx']))]:
        return'correct'
    else:
        return 'incorrect'

df = df.loc[df['Dataset'].isin(datasets)]

df[f'{model_name} zero-shot prediction'] = df.apply(add_correct_fn, axis=1)
df['Dataset'] = df.apply(update_dataset_display_name, axis=1)

datasets_in_order = sorted(list(set(df['Dataset'].tolist())))

palette = {'correct': '#8da0cb',
           'incorrect': '#fc8d62'}

f = plt.figure(figsize=(19.2, 3.6))
ax = plt.axes()
sns.violinplot(data=df, ax=ax, y=f'{cosine_sim_field} cosine similarity', x='Dataset', hue=f'{model_name} zero-shot prediction', palette=palette, orient='v', showfliers=False, zorder=10, clip_on=False, order=datasets_in_order, hue_order=['correct', 'incorrect'], width=0.7, cut=0)
ax.set_ylabel(f'{cosine_sim_label} cosine similarity', fontsize=18)
ax.set_xlabel('Dataset', fontsize=18)
ax.set_ylim((ymin,1))
ax.grid(visible=True, axis='y', which='minor')
ax.tick_params(axis='y', which='major', labelsize=14)
ax.tick_params(axis='x', which='major', labelsize=10)
ax.set_clip_on(False)
for tick in ax.xaxis.get_major_ticks()[1::2]:
    tick.set_pad(20)
legend = plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title=f'{model_name} zero-shot prediction')
savefig(f'./figures/bigbench-{model}-violinplots_{cosine_sim}-cosine.pdf', bbox_inches='tight')
