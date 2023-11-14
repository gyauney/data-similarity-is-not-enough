from collections import defaultdict
import numpy as np
import pandas as pd
# don't let matplotlib use xwindows
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pylab import savefig
import seaborn as sns
sns.set_style("whitegrid")
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--reproduction', action=argparse.BooleanOptionalAction)
args = parser.parse_args()

# Default values for reading newly generated results
xnli_csv_filename = './xnli-output-pythia-6.9b/df_percentages_partial.csv'
# Values for reproducing figure 2 from cached raw results
if args.reproduction:
    xnli_csv_filename = './raw-results/xnli_pythia-6.9b_partial-languages.csv'

output_dir = './figures'
xnli_df = pd.read_csv(xnli_csv_filename)
xnli_languages = xnli_df['Language'].tolist()
xnli_df['Base dataset'] = 'XNLI'
xnli_to_cat = xnli_df[['Base dataset', 'Language', 'KL-divergence from the Pile (Pythia unigrams)', 'KL-divergence from the Pile (Pythia unigrams) (low CI)', 'KL-divergence from the Pile (Pythia unigrams) (high CI)', 'Normalized score (LM)', 'Percentage']]

languages = list(set(xnli_languages))
df = pd.concat([xnli_to_cat], ignore_index=True)

language_to_display_name = {
    'english': 'English',
    'finnish': 'Finnish',
    'french': 'French',
    'swahili': 'Swahili',
    'indonesian': 'Indonesian',
    'spanish': 'Spanish',
    'german': 'German',
    'bulgarian': 'Bulgarian',
    'russian': 'Russian',
    'turkish': 'Turkish',
    'arabic': 'Arabic',
    'vietnamese': 'Vietnamese',
    'greek': 'Greek',
    'thai': 'Thai',
    'chinese': 'Chinese',
    'hindi': 'Hindi',
    'urdu': 'Urdu',
}

dx = 12
dxx = 6
dy = 12
language_to_text_offset = {
    'english': (-dx, -2),
    'finnish': (dxx, -dy),
    'french': (dx, 0),
    'swahili': (0, 15),
    'indonesian': (dx, 0),
    'spanish': (dx, 0),
    'german': (dx, 0),
    'bulgarian': (dx, 0),
    'russian': (dx, 0),
    'turkish': (dx, 0),
    'arabic': (dx, 0),
    'vietnamese': (dx, 0),
    'greek':  (dx, 0),
    'thai':  (dx, 0),
    'chinese': (dx, 0),
    'hindi':  (dx, 0),
    'urdu':  (dx, 0),
}

language_to_ha = {
    'english': 'right',
    'finnish': 'left',
    'french': 'left',
    'swahili': 'left',
    'indonesian': 'left',
    'spanish': 'left',
    'german': 'left',
    'bulgarian': 'left',
    'russian': 'left',
    'turkish': 'left',
    'arabic': 'left',
    'vietnamese': 'left',
    'greek':  'left',
    'thai':  'left',
    'chinese': 'left',
    'hindi':  'left',
    'urdu':  'left',
}

# just boring blue
color = np.array((0.3215686274509804, 0.4470588235294118, 0.6705882352941176))
english_color = '#eeeeee'

f, axs = plt.subplots(2, 7, figsize=(18, 3))
plt.subplots_adjust(hspace=0.6, wspace=0.25)

languages_to_plot = ['french', 'swahili', 'spanish', 'german', 'bulgarian', 'russian', 'turkish', 'arabic', 'vietnamese', 'greek', 'thai', 'chinese', 'hindi', 'urdu']
df_to_plot = df.loc[df['Base dataset'] == 'XNLI']
for i, language in enumerate(sorted(languages_to_plot)):
    ax = axs[math.floor(i / 7), i % 7]
    ax.grid(alpha=0.6)
    df_just_lang = df_to_plot.loc[df_to_plot['Language'] == language]
    x = df_just_lang['KL-divergence from the Pile (Pythia unigrams)'].tolist()
    y = df_just_lang['Normalized score (LM)'].tolist()
    ps = df_just_lang['Percentage'].tolist()
    ps = [p + 25 for p in ps]
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    cs = [(1 - p) * np.array([1, 1, 1]) + p * color for p in [0.25, 0.5, 0.75, 1]]
    cmap = ListedColormap(cs)

    lc = LineCollection(segments, cmap=cmap, capstyle='round')
    #lc.set_array(ps)
    lc.set_array([0, 1, 2, 3])
    lc.set_linewidth(5)
    ax.add_collection(lc)
    for _, row in df_just_lang.iterrows():
        if row['Percentage'] not in [100]:
            continue
        if row['Language'] == 'english':
            continue
        s = 150
        linewidth = 1
        alpha = 1
        z = 20
        l = row['Language']
        p = row['Percentage']/100
        c = (1 - p) * np.array([1, 1, 1]) + p * color
        ax.scatter([row["KL-divergence from the Pile (Pythia unigrams)"]], [row["Normalized score (LM)"]], s=s, alpha=alpha, color=c, linewidth=linewidth, edgecolor='#ffffff', zorder=z, clip_on=False)
    # plot the english dot
    row = df_to_plot.loc[(df_to_plot['Language'] == 'english') & (df_to_plot['Percentage'] == 100)]
    ax.scatter([row["KL-divergence from the Pile (Pythia unigrams)"]], [row["Normalized score (LM)"]], s=150, alpha=1, color=english_color, edgecolor='#cccccc', zorder=20)
    l = 'english'
    ax.plot([row["KL-divergence from the Pile (Pythia unigrams) (low CI)"], row["KL-divergence from the Pile (Pythia unigrams) (high CI)"]], [row["Normalized score (LM)"], row["Normalized score (LM)"]],
                linewidth=12, color='#cccccc', alpha=0.1, clip_on=False,
                zorder=1)
    # add error bars to whole line!!!
    ax.fill_betweenx(y=df_just_lang['Normalized score (LM)'], x1=df_just_lang["KL-divergence from the Pile (Pythia unigrams) (low CI)"], x2=df_just_lang["KL-divergence from the Pile (Pythia unigrams) (high CI)"], zorder=1, clip_on=True, alpha=0.15, color=color, linewidth=8)
    ax.set_xlim(0.5, 8)
    ax.set_ylim(0, 18)
    ax.set_xticks([0,2,4,6,8])
    ax.set_yticks([0,5,10,15,20])
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
    ax.set_title(language_to_display_name[language], fontsize=14, fontweight='bold')
plt.text(0.1, 0.5, 'Normalized score', fontsize=14, transform=f.transFigure, ha='center', va='center', rotation=90)
plt.text(0.52, -0.04, 'KL-divergence from pretraining token distribution', fontsize=14, transform=f.transFigure, ha='center', va='center')
    
savefig(f'{output_dir}/figure-5.pdf', bbox_inches='tight')
plt.close()

for language in ['french', 'swahili', 'spanish', 'german', 'bulgarian', 'russian', 'turkish', 'arabic', 'vietnamese', 'greek', 'thai', 'chinese', 'hindi', 'urdu']:
    df_to_test = df_to_plot.loc[(df_to_plot['Language'] == language) & (df_to_plot['Percentage'] > 0)]
    x = df_to_test['KL-divergence from the Pile (Pythia unigrams)'].tolist()
    y = df_to_test['Normalized score (LM)'].tolist()