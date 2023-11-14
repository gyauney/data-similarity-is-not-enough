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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--reproduction', action=argparse.BooleanOptionalAction)
args = parser.parse_args()

# Default values for reading newly generated results
stackexchange_csv_filename = './stackexchange-output-pythia-6.9b_forum-labels/df_percentages_partial.csv'
xnli_csv_filename = './xnli-output-pythia-6.9b/df_percentages_partial.csv'
# Values for reproducing figure 2 from cached raw results
if args.reproduction:
    stackexchange_csv_filename = './raw-results/stackexchange_pythia-6.9b_partial-languages.csv'
    xnli_csv_filename = './raw-results/xnli_pythia-6.9b_partial-languages.csv'

output_dir = './figures'

stackexchange_df = pd.read_csv(stackexchange_csv_filename)
stackexchange_languages = stackexchange_df['Language'].tolist()
stackexchange_df['Base dataset'] = 'Stackexchange'
stackexchange_to_cat = stackexchange_df[['Base dataset', 'Language', 'KL-divergence from the Pile (Pythia unigrams)', 'KL-divergence from the Pile (Pythia unigrams) (low CI)', 'KL-divergence from the Pile (Pythia unigrams) (high CI)', 'Normalized score (LM)', 'Percentage']]

# original for submission:
xnli_df = pd.read_csv(xnli_csv_filename)
#xnli_languages = xnli_df['Language'].tolist()
xnli_languages = ['english', 'french', 'german', 'spanish', 'swahili']
xnli_languages_to_plot = set(['english', 'french', 'german', 'spanish', 'swahili'])
xnli_df['Base dataset'] = 'XNLI'
xnli_to_cat = xnli_df[['Base dataset', 'Language', 'KL-divergence from the Pile (Pythia unigrams)', 'KL-divergence from the Pile (Pythia unigrams) (low CI)', 'KL-divergence from the Pile (Pythia unigrams) (high CI)', 'Normalized score (LM)', 'Percentage']]
# put spanish last so drawn on top in the LineCollection
xnli_without_spanish = xnli_to_cat.loc[xnli_to_cat['Language'] != 'spanish']
xnli_with_spanish = xnli_to_cat.loc[xnli_to_cat['Language'] == 'spanish']

languages = list(set(stackexchange_languages + xnli_languages))
df = pd.concat([stackexchange_to_cat, xnli_with_spanish, xnli_without_spanish], ignore_index=True)

dataset_to_color = {'english': (0.9677975592919913, 0.44127456009157356, 0.5358103155058701), 'finnish': (0.7757319041862729, 0.5784925270759935, 0.19475566538551875), 'spanish': (0.5105309046900421, 0.6614299289084904, 0.1930849118538962), 'french': (0.20433460114757862, 0.6863857739476534, 0.5407103379425205), 'german': (0.21662978923073606, 0.6676586160122123, 0.7318695594345369), 'indonesian': (0.5049017849530067, 0.5909119231215284, 0.9584657252128558), 'swahili': (0.9587050080494409, 0.3662259565791742, 0.9231469575614251)}
english_color = '#eeeeee'

f, axs = plt.subplots(2, 1, figsize=(6.4, 6))
plt.subplots_adjust(hspace=0.5)
ax = axs[0]
ax.grid(alpha=0.6)
df_to_plot = df.loc[df['Base dataset'] == 'Stackexchange']
#g = sns.lineplot(data=df_to_plot, ax=ax, x='KL-divergence from the Pile (Pythia unigrams)', y='Normalized score (LM)', hue='Language', palette=dataset_to_color, linewidth=4, alpha=0.7, clip_on=False, zorder=10, sort=False)

language_to_display_name = {
    'english': 'English',
    'finnish': 'Finnish',
    'french': 'French',
    'swahili': 'Swahili',
    'indonesian': 'Indonesian',
    'spanish': 'Spanish',
    'german': 'German',
}

dx = 12
dxx = 6
dy = 12
language_to_text_offset = {
    'english': (dx, 6),
    'finnish': (dxx, -dy),
    'french': (dx, 0),
    'swahili': (dx, 0),
    'indonesian': (dx, 0),
    'spanish': (dxx, dy),
    'german': (dx, 0),
}

language_to_ha = {
    'english': 'left',
    'finnish': 'left',
    'french': 'left',
    'swahili': 'left',
    'indonesian': 'left',
    'spanish': 'left',
    'german': 'left',
}

for language in languages:
    if language == 'english' or language == 'greek':
        continue
    df_just_lang = df_to_plot.loc[df_to_plot['Language'] == language]
    x = df_just_lang['KL-divergence from the Pile (Pythia unigrams)'].tolist()
    y = df_just_lang['Normalized score (LM)'].tolist()
    ps = df_just_lang['Percentage'].tolist()
    ps = [p for p in ps]
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    cs = [(1 - p) * np.array([1, 1, 1]) + p * np.array(dataset_to_color[language]) for p in [0.25, 0.5, 0.75, 1]]
    cmap=ListedColormap(cs)
    
    lc = LineCollection(segments, cmap=cmap, capstyle='round')
    #lc.set_array(ps)
    lc.set_array([0, 1, 2, 3])
    lc.set_linewidth(5)
    ax.add_collection(lc)
for _, row in df_to_plot.iterrows():
    if row['Percentage'] not in [100]:
       continue
    if row['Language'] == 'english':
        continue
    s = 150
    linewidth = 1
    alpha = 1
    l = row['Language']
    p = row['Percentage']/100
    c = (1 - p) * np.array([1, 1, 1]) + p * np.array(dataset_to_color[l])
    ax.scatter([row["KL-divergence from the Pile (Pythia unigrams)"]], [row["Normalized score (LM)"]], s=s, alpha=alpha, color=c, edgecolor='#ffffff', linewidth=linewidth, zorder=20)
    ax.annotate(language_to_display_name[l],
                #color='#555555',
                fontweight='light',
                xy=(row["KL-divergence from the Pile (Pythia unigrams)"], row["Normalized score (LM)"]), xycoords='data',
                xytext=language_to_text_offset[l], textcoords='offset points',
                ha=language_to_ha[l], va='center',
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.1",
                      fc=dataset_to_color[l], lw=0, alpha=0.1))
# plot the english dot
row = df_to_plot.loc[(df_to_plot['Language'] == 'english') & (df_to_plot['Percentage'] == 100)]
if len(row):
    ax.scatter([row["KL-divergence from the Pile (Pythia unigrams)"]], [row["Normalized score (LM)"]], s=150, alpha=1, color=english_color, edgecolor='#cccccc', zorder=20)
    l = 'english'
    ax.annotate(language_to_display_name[l],
                #color='#555555',
                fontweight='light',
                xy=(row["KL-divergence from the Pile (Pythia unigrams)"], row["Normalized score (LM)"]), xycoords='data',
                xytext=language_to_text_offset[l], textcoords='offset points',
                ha=language_to_ha[l], va='center',
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.1",
                        fc='#aaaaaa', lw=0, alpha=0.15))
    ax.plot([row["KL-divergence from the Pile (Pythia unigrams) (low CI)"], row["KL-divergence from the Pile (Pythia unigrams) (high CI)"]], [row["Normalized score (LM)"], row["Normalized score (LM)"]],
            linewidth=12, color='#cccccc', alpha=0.1, clip_on=False,
            zorder=1)#, solid_capstyle='projecting')
# add error bars to whole line!!!
for l in ['finnish', 'french', 'swahili', 'indonesian', 'spanish', 'german']:
    to_plot = df_to_plot.loc[(df_to_plot['Language'] == l)]
    ax.fill_betweenx(y=to_plot['Normalized score (LM)'], x1=to_plot["KL-divergence from the Pile (Pythia unigrams) (low CI)"], x2=to_plot["KL-divergence from the Pile (Pythia unigrams) (high CI)"], zorder=1, clip_on=False, alpha=0.15, color=dataset_to_color[l], linewidth=12)

ax.set_ylim(30, 82)
#ax.get_legend().remove()
ax.set_title('Stack Exchange forum classification', fontsize=14, fontweight='bold')
# ax.set_ylim(30, 80)
ax.set_xlim(0.5, 4)
ax.set_xlabel('KL-divergence from pretraining token distribution', fontsize=14)
ax.set_ylabel('Normalized score', fontsize=14)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)


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
}

language_to_ha = {
    'english': 'right',
    'finnish': 'left',
    'french': 'left',
    'swahili': 'left',
    'indonesian': 'left',
    'spanish': 'left',
    'german': 'left',
}



df_to_plot = df.loc[df['Base dataset'] == 'XNLI']
ax = axs[1]
ax.grid(alpha=0.6)
#sns.lineplot(data=df_to_plot, ax=ax, x='KL-divergence from the Pile (Pythia unigrams)', y='Normalized score (LM)', hue='Language', palette=dataset_to_color, linewidth=4, alpha=0.7, clip_on=False, zorder=10, sort=False)
for language in languages:
    if language == 'english' or language == 'greek':
        continue
    df_just_lang = df_to_plot.loc[df_to_plot['Language'] == language]
    x = df_just_lang['KL-divergence from the Pile (Pythia unigrams)'].tolist()
    y = df_just_lang['Normalized score (LM)'].tolist()
    ps = df_just_lang['Percentage'].tolist()
    ps = [p + 25 for p in ps]
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    cs = [(1 - p) * np.array([1, 1, 1]) + p * np.array(dataset_to_color[language]) for p in [0.25, 0.5, 0.75, 1]]
    cmap = ListedColormap(cs)

    lc = LineCollection(segments, cmap=cmap, capstyle='round')
    lc.set_array([0, 1, 2, 3])
    lc.set_linewidth(5)
    ax.add_collection(lc)
for _, row in df_to_plot.iterrows():
    if row['Percentage'] not in [100]:
        continue
    if row['Language'] == 'english':
        continue
    if row['Language'] not in xnli_languages_to_plot:
        continue
    s = 150
    linewidth = 1
    alpha = 1
    z = 20
    l = row['Language']
    p = row['Percentage']/100
    c = (1 - p) * np.array([1, 1, 1]) + p * np.array(dataset_to_color[l])
    ax.scatter([row["KL-divergence from the Pile (Pythia unigrams)"]], [row["Normalized score (LM)"]], s=s, alpha=alpha, color=c, linewidth=linewidth, edgecolor='#ffffff', zorder=z)
    ax.annotate(language_to_display_name[l],
                fontweight='light',
                xy=(row["KL-divergence from the Pile (Pythia unigrams)"], row["Normalized score (LM)"]), xycoords='data',
                xytext=language_to_text_offset[l], textcoords='offset points',
                ha=language_to_ha[l], va='center',
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.1",
                      fc=dataset_to_color[l], lw=0, alpha=0.1))
# plot the english dot
row = df_to_plot.loc[(df_to_plot['Language'] == 'english') & (df_to_plot['Percentage'] == 100)]
ax.scatter([row["KL-divergence from the Pile (Pythia unigrams)"]], [row["Normalized score (LM)"]], s=150, alpha=1, color=english_color, edgecolor='#cccccc', zorder=20)
l = 'english'
ax.annotate(language_to_display_name[l],
            fontweight='light',
            xy=(row["KL-divergence from the Pile (Pythia unigrams)"], row["Normalized score (LM)"]), xycoords='data',
            xytext=language_to_text_offset[l], textcoords='offset points',
            ha=language_to_ha[l], va='center',
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.1",
                    fc='#aaaaaa', lw=0, alpha=0.15))
ax.plot([row["KL-divergence from the Pile (Pythia unigrams) (low CI)"], row["KL-divergence from the Pile (Pythia unigrams) (high CI)"]], [row["Normalized score (LM)"], row["Normalized score (LM)"]],
            linewidth=12, color='#cccccc', alpha=0.1, clip_on=False,
            zorder=1)#, solid_capstyle='projecting')
# add error bars to whole line!!!
for l in ['french', 'swahili', 'spanish', 'german']:
    to_plot = df_to_plot.loc[(df_to_plot['Language'] == l)]
    ax.fill_betweenx(y=to_plot['Normalized score (LM)'], x1=to_plot["KL-divergence from the Pile (Pythia unigrams) (low CI)"], x2=to_plot["KL-divergence from the Pile (Pythia unigrams) (high CI)"], zorder=1, clip_on=False, alpha=0.15, color=dataset_to_color[l], linewidth=12)
ax.set_title('XNLI', fontsize=14, fontweight='bold')
ax.set_xlim(0.5, 4)
ax.set_xlabel('KL-divergence from pretraining token distribution', fontsize=14)
ax.set_ylabel('Normalized score', fontsize=14)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
savefig(f'{output_dir}/figure-2.pdf', bbox_inches='tight')
plt.close()

for language in ['spanish', 'french', 'swahili', 'german']:
    df_to_test = df_to_plot.loc[(df_to_plot['Language'] == language) & (df_to_plot['Percentage'] > 0)]
    x = df_to_test['KL-divergence from the Pile (Pythia unigrams)'].tolist()
    y = df_to_test['Normalized score (LM)'].tolist()