import numpy as np
import pandas as pd
import os
# don't let matplotlib use xwindows
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pylab import savefig
import seaborn as sns
sns.set_style("whitegrid")
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--reproduction', action=argparse.BooleanOptionalAction)
args = parser.parse_args()

# Default values for reading newly generated results
bigbench_csv_filename = './bigbench-output-pythia-6.9b/df_bigbench_with_similarities.csv'
bigbench_similarity_col = 'KL-divergence from the Pile (Pythia unigrams)'
bigbench_perf_col = 'Normalized score (LM)'
# Values for reproducing figure 1 from cached raw results because
# there are a few naming differences
if args.reproduction:
    bigbench_csv_filename = './raw-results/bigbench-raw-results-multiple-choice.csv'
    bigbench_similarity_col = 'Average KL-divergence from the Pile (Pythia unigrams)'
    bigbench_perf_col = 'Pythia 6.9B (0 shot) normalized aggregate score'

output_dir = './figures'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

bigbench_df = pd.read_csv(bigbench_csv_filename)
bigbench_df['Base dataset'] = 'BIG-bench Lite'
bigbench_to_cat = bigbench_df[['Base dataset', 'Dataset', bigbench_similarity_col, 'KL-divergence from the Pile (Pythia unigrams) (low CI)', 'KL-divergence from the Pile (Pythia unigrams) (high CI)', bigbench_perf_col]]

x = bigbench_df[bigbench_similarity_col].tolist()
y = bigbench_df[bigbench_perf_col].tolist()

stackexchange_df = pd.read_csv('./raw-results/stackexchange_pythia-6.9b_full-languages.csv')
languages = stackexchange_df['Language'].tolist()
stackexchange_df['Dataset'] = stackexchange_df['Language']
stackexchange_df['Base dataset'] = 'Stackexchange'
stackexchange_to_cat = stackexchange_df[['Base dataset', 'Dataset', 'KL-divergence from the Pile (Pythia unigrams)', 'KL-divergence from the Pile (Pythia unigrams) (low CI)', 'KL-divergence from the Pile (Pythia unigrams) (high CI)', 'Normalized score (LM)']]

df = pd.concat([bigbench_to_cat, stackexchange_to_cat])

colors = sns.husl_palette(len(languages))
dataset_to_color = {l: c for l, c in zip(languages, colors)}

ddx = 15
dx = 12
dy = 7
language_to_text_offset = {
    'english': (dx, 0),
    'finnish': (-dx, 0),
    'french': (dx, 0),
    'swahili': (dx, 0),
    'indonesian': (dx, 0),
    'spanish': (-ddx, 0),
    'german': (-ddx, 0),
}

language_to_ha = {
    'english': 'left',
    'finnish': 'right',
    'french': 'left',
    'swahili': 'left',
    'indonesian': 'left',
    'spanish': 'right',
    'german': 'right',
}


language_to_display_name = {
    'english': 'English',
    'finnish': 'Finnish',
    'french': 'French',
    'swahili': 'Swahili',
    'indonesian': 'Indonesian',
    'spanish': 'Spanish',
    'german': 'German',
}

f, axs = plt.subplots(2, 1, figsize=(6.4, 6))
plt.subplots_adjust(hspace=0.5)
ax = axs[0]
ax.grid(alpha=0.6)
df_to_plot = df.loc[df['Base dataset'] == 'Stackexchange']
g = sns.scatterplot(data=df_to_plot, ax=ax, x='KL-divergence from the Pile (Pythia unigrams)', y='Normalized score (LM)', hue='Dataset', palette=dataset_to_color, linewidth=0, s=150, alpha=0.9, clip_on=False, zorder=10)
ax.get_legend().remove()
ax.set_title('Stack Exchange forum classification', fontsize=14, fontweight='bold')
for _, row in df_to_plot.iterrows():
    l = row['Dataset']
    ax.annotate(language_to_display_name[l],
                #color='#555555',
                fontweight='light',
                xy=(row["KL-divergence from the Pile (Pythia unigrams)"], row["Normalized score (LM)"]), xycoords='data',
                xytext=language_to_text_offset[l], textcoords='offset points',
                ha=language_to_ha[l], va='center',
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.1",
                      fc=dataset_to_color[l], lw=0, alpha=0.15))
    ax.plot([row["KL-divergence from the Pile (Pythia unigrams) (low CI)"], row["KL-divergence from the Pile (Pythia unigrams) (high CI)"]], [row["Normalized score (LM)"], row["Normalized score (LM)"]],
            linewidth=12, color='#cccccc', alpha=0.7,
            zorder=2)#, solid_capstyle='projecting')
ax.set_ylim(30, 80)
ax.set_xlim(0.5, 4)
ax.set_xlabel('KL-divergence from pretraining token distribution', fontsize=14)
ax.set_ylabel('')
ax.text(-0.12, 0.5, 'Normalized score', fontsize=14, rotation=90, ha='center', va='center', transform=ax.transAxes)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
h1, l1 = ax.get_legend_handles_labels()


df_to_plot = df.loc[df['Base dataset'] == 'BIG-bench Lite']
ax = axs[1]
ax.grid(alpha=0.6)
c = '#b3cde3'
sns.scatterplot(data=df_to_plot, ax=ax, x=bigbench_similarity_col, y=bigbench_perf_col, color=c, linewidth=0, s=150, alpha=0.9, clip_on=False, zorder=10)
for _, row in df_to_plot.iterrows():
    ax.plot([row["KL-divergence from the Pile (Pythia unigrams) (low CI)"], row["KL-divergence from the Pile (Pythia unigrams) (high CI)"]], [row[bigbench_perf_col], row[bigbench_perf_col]],
            linewidth=12, color='#cccccc', alpha=0.7, clip_on=False,
            zorder=2)
ax.set_title('BIG-bench Lite multiple choice tasks', fontsize=14, fontweight='bold')
ax.set_xlim(0.5, 4)
ax.set_xlabel('KL-divergence from pretraining token distribution', fontsize=14)
ax.set_ylabel('')
ax.text(-0.12, 0.5, 'Normalized score', fontsize=14, rotation=90, ha='center', va='center', transform=ax.transAxes)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
savefig(f'{output_dir}/figure-1.pdf', bbox_inches='tight')
plt.close()


