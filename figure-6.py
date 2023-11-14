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
from scipy.stats import bootstrap
import functools

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=False, default='Pythia-6.9B', type=str)
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

dataset_to_random_guess_accuracy = {
    "hindu_knowledge": 0.2480952380952381,
    "bbq_lite_json": 0.3333333333333082,
    "code_line_description": 0.24833333333333335,
    "conceptual_combinations": 0.25,
    "emoji_movie": 0.19999999999999962,
    "formal_fallacies": 0.5,
    "known_unknowns": 0.5,
    "language_identification": 0.0909090909090978,
    "logic_grid_puzzle": 0.3108999999999984,
    "logical_deduction": 0.1999999999999981,
    "misconceptions_russian": 0.5,
    "novel_concepts": 0.1937500000000001,
    "play_dialog_same_or_different": 0.5,
    "strange_stories": 0.3261494252873563,
    "strategyqa": 0.5,
    "symbol_interpretation": 0.19999999999999726,
    "vitaminc_fact_verification": 0.33333333333342896,
    "winowhy": 0.5
}

def normalize_score(dataset, accuracy):
    # All tasks have a max accuracy of 1
    low_score = dataset_to_random_guess_accuracy[dataset]
    high_score = 1.0
    return 100 * (accuracy - low_score) / (high_score - low_score)


df = pd.read_csv(f'./raw-results/cosine-similarities/max-cosine_bigbench_the-pile.csv')
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

def add_correct_int_fn(row):
    if row['Dataset'] not in datasets:
        return 0
    if dataset_to_index_to_correct[row['Dataset']][str(int(row['Idx']))]:
        return 1
    else:
        return 0

df = df.loc[df['Dataset'].isin(datasets)]

df[f'{model_name} zero-shot prediction'] = df.apply(add_correct_fn, axis=1)
df[f'{model_name} zero-shot prediction int'] = df.apply(add_correct_int_fn, axis=1)
df['Dataset'] = df.apply(update_dataset_display_name, axis=1)

datasets_in_order = sorted(list(set(df['Dataset'].tolist())))

number_to_quartile_name = {
    1: 'First (least similar)',
    2: 'Second',
    3: 'Third',
    4: 'Fourth (most similar)',
}

def correct_to_norm_score(correct, dataset):
    acc = sum(correct) / len(correct)
    return normalize_score(dataset, acc)


data_binned_by_similarity = []
for dataset in datasets_in_order:
    df_dataset = df.loc[df['Dataset'] == dataset]
    vs = np.quantile(df_dataset['Max cosine similarity'],
                            [0, 0.25, 0.5, 0.75, 1])
    
    print(dataset)
    for q, (v1, v2) in enumerate(zip(vs, vs[1:])):
        this_df = df_dataset.loc[(df_dataset['Max cosine similarity'] >= v1) & (df_dataset['Max cosine similarity'] <= v2)]
        correct = this_df[f'{model_name} zero-shot prediction int']
        acc = sum(correct) / len(correct)
        norm_score_fn = functools.partial(correct_to_norm_score, dataset=dataset)
        res = bootstrap((correct,), norm_score_fn)
        for norm_score in res.bootstrap_distribution:
            data_binned_by_similarity.append({'Dataset': dataset, 
                                            'Normalized score': norm_score,
                                            'Quartile of similarity': number_to_quartile_name[q+1],
                                            })

df_binned_by_similarity = pd.DataFrame(data_binned_by_similarity)

palette = {
    'First (least similar)': '#c6dbef',
    'Second': '#9ecae1',
    'Third': '#6baed6',
    'Fourth (most similar)': '#3182bd',
}

f = plt.figure(figsize=(19.2, 3.2))
ax = plt.axes()
sns.barplot(data=df_binned_by_similarity, ax=ax, y='Normalized score', x='Dataset', errorbar='sd', hue='Quartile of similarity', palette=palette, orient='v', err_kws={'color': '#aaaaaa99', 'linewidth': 1}, order=datasets_in_order, hue_order=['First (least similar)', 'Second', 'Third', 'Fourth (most similar)'], width=0.7)
ax.set_ylabel('Normalized score', fontsize=18)
ax.set_xlabel('Dataset', fontsize=18)
ax.tick_params(axis='y', which='major', labelsize=14)
ax.tick_params(axis='x', which='major', labelsize=10)
ax.set_clip_on(False)
for tick in ax.xaxis.get_major_ticks()[1::2]:
    tick.set_pad(20)
legend = plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='Quartile of similarity')
savefig(f'./figures/bigbench-{model}-similarity-quartiles-barplot.pdf', bbox_inches='tight')
