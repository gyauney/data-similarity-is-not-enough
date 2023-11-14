import json
import numpy as np
import pandas as pd
# don't let matplotlib use xwindows
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pylab import savefig
import seaborn as sns
sns.set_style("whitegrid")

plt.rcParams['legend.fontsize'] = 14
plt.rcParams['legend.title_fontsize'] = 14

dataset_to_print = {'mnli_matched': 'MNLI\nmatched',
                    'mnli_mismatched': 'MNLI\nmismatched',
                    'qnli': 'QNLI',
                    'qqp': 'QQP',
                    'sst2': 'SST2',
                    'rte': 'RTE'}
datasets_in_order = ['MNLI\nmatched', 'MNLI\nmismatched', 'QNLI', 'QQP', 'SST2', 'RTE']

df_pretrain = pd.read_csv('./raw-results/cosine-similarities/max-cosine_glue-validation_c4.csv')
df_pretrain['Dataset'] = df_pretrain['Dataset'].apply(lambda d: dataset_to_print[d])

df_train = pd.read_csv('./raw-results/cosine-similarities/max-cosine_glue-validation_glue-train.csv')
df_train['Dataset'] = df_train['Dataset'].apply(lambda d: dataset_to_print[d])

palette = {'correct': '#8da0cb',
           'incorrect': '#fc8d62'}

f, axs = plt.subplots(1, 2, figsize=(19.2, 4.8))
sns.violinplot(data=df_pretrain, ax=axs[0], y='Max cosine similarity', x='Dataset', hue='Finetuned T5 prediction', palette=palette, orient='v', showfliers=False, zorder=10, order=datasets_in_order, hue_order=['correct', 'incorrect'], width=0.7, cut=0, clip_on=False)
sns.violinplot(data=df_train, ax=axs[1], y='Max cosine similarity', x='Dataset', hue='Finetuned T5 prediction', palette=palette, orient='v', showfliers=False, zorder=10, order=datasets_in_order, hue_order=['correct', 'incorrect'], width=0.7, cut=0, clip_on=False)
axs[1].get_legend().remove()
axs[0].set_ylabel('Max cosine similarity', fontsize=20)
axs[1].set_ylabel('')
for ax in axs:
    ax.set_ylim((0.83,1))
    ax.set_yticks([0.85, 0.9, 0.95, 1])
    ax.grid(visible=True, axis='y', which='minor')
    ax.minorticks_on()
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.025))
    ax.set_xlabel('Dataset', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_clip_on(False)
axs[0].set_title('Similarity to pretraining data', fontsize=22, fontweight='bold')
axs[1].set_title('Similarity to finetuning data', fontsize=22, fontweight='bold')
savefig(f'./figures/glue-t5-violinplots_max-cosine.pdf', bbox_inches='tight')
