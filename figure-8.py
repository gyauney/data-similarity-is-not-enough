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

output_dir = './figures'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df = pd.read_csv('./raw-results/stackexchange_pythia-6.9b_ampm-labels.csv')
languages = df['Language'].tolist()
colors = sns.husl_palette(len(languages))
dataset_to_color = {l: c for l, c in zip(languages, colors)}

ddx = 15
dx = 12
dy = 10
language_to_text_offset = {
    'english': (10, dy),
    'finnish': (17.5, dy),
    'french': (-30, -dy),
    'swahili': (20, dy),
    'indonesian': (33, -dy),
    'spanish': (-10, dy),
    'german': (8, -dy),
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

language_to_va = {
    'english': 'bottom',
    'finnish': 'bottom',
    'french': 'top',
    'swahili': 'bottom',
    'indonesian': 'top',
    'spanish': 'bottom',
    'german': 'top',
}

f = plt.figure(figsize=(6.4, 2.2))
ax = plt.axes()
ax.grid(alpha=0.6)
g = sns.scatterplot(data=df, ax=ax, x='KL-divergence from the Pile (Pythia unigrams)', y='Normalized score (LM)', hue='Language', palette=dataset_to_color, linewidth=0, s=150, alpha=0.9, clip_on=False, zorder=10)
ax.get_legend().remove()
ax.set_title('Stack Exchange AM/PM classification', fontsize=14, fontweight='bold')
for _, row in df.iterrows():
    l = row['Language']
    ax.annotate(language_to_display_name[l],
                fontweight='light',
                xy=(row["KL-divergence from the Pile (Pythia unigrams)"], row["Normalized score (LM)"]), xycoords='data',
                xytext=language_to_text_offset[l], textcoords='offset points',
                ha='center', va=language_to_va[l],
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.1",
                      fc=dataset_to_color[l], lw=0, alpha=0.15))
    ax.plot([row["KL-divergence from the Pile (Pythia unigrams) (low CI)"], row["KL-divergence from the Pile (Pythia unigrams) (high CI)"]], [row["Normalized score (LM)"], row["Normalized score (LM)"]],
            linewidth=12, color='#cccccc', alpha=0.7,
            zorder=2)
ax.set_ylim(-5, 20)
ax.set_xlim(0.5, 4)
ax.set_xlabel('KL-divergence from pretraining token distribution', fontsize=14)
ax.set_ylabel('Normalized score', fontsize=14)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
savefig(f'{output_dir}/figure-8.pdf', bbox_inches='tight')
plt.close()