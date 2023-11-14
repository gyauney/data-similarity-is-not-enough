import json
from collections import defaultdict
import numpy as np
import random
import pandas as pd
import os
from tqdm import tqdm
import operator
# don't let matplotlib use xwindows
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pylab import savefig
import seaborn as sns
sns.set_style("whitegrid")
from scipy.special import rel_entr
import mauve
from scipy.stats import bootstrap

base_dataset = 'bicycles-cstheory'

def normalize_score(acc):
    low = 0.5
    return 100 * (acc - low) / (1 - low)

eps = 1e-8
vocab_size = 51000

def vectorify(token_to_counts):
    v = np.zeros(vocab_size, dtype=np.float32)
    for token, count in tqdm(token_to_counts.items()):
        v[int(token)] += count
    return v

def normalize_dist(v):
    return v / np.sum(v)

def smooth(d):
    d += eps
    return normalize_dist(d)

print('Getting pretraining Pythia token distributions.')
num_pile_samples = 8
pretraining_dists_pythia = []
for sample_num in range(1, num_pile_samples + 1):
    with open(f'./n-gram-frequencies/the_pile-train_100000-docs_sample-{sample_num}_frequencies-unigram-tokens.json', 'r') as f:
        pretraining_ngram_to_counts = json.load(f)
    pretraining_dists_pythia.append(smooth(normalize_dist(vectorify(pretraining_ngram_to_counts))))

results_dir = './stackexchange-output-pythia-6.9b_ampm-labels'
output_dir = './figures'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def get_only_idxs(A, idxs):
    return [A[idx] for idx in idxs]

idx_to_correct = defaultdict(dict)
idx_to_correct_pmi_dc = defaultdict(dict)
idx_to_correct_target_perplexities = defaultdict(dict)
idx_to_correct_target_probabilities = defaultdict(dict)
idx_to_input_perplexities = defaultdict(dict)

idx_to_wpt_ngram_counts_hashed = defaultdict(dict)
idx_to_pythia_unigram_counts = defaultdict(dict)

langauge_to_max_sims_in_order = {}
language_to_embeddings_in_order = {}

languages = ['finnish', 'spanish', 'french', 'german', 'indonesian', 'swahili']
for language in ['english'] + languages:
  print(language)
  dataset = f'{base_dataset}_100-percent-{language}_ampm-labels-prompted'
  results_dataset = f'{base_dataset}_100-percent-{language}'

  # first load token distributions
  # Pythia tokens
  with open(f'./data-stackexchange/{dataset}_pythia-2.8b-idx-to-counts-unigram-tokens.json', 'r') as f:
    idx_to_counts = json.load(f)
  for idx, token_to_counts in idx_to_counts.items():
    idx_to_pythia_unigram_counts[language][idx] = vectorify(token_to_counts)

  example_to_choices = defaultdict(list)
  example_to_choices_pmi_dc = defaultdict(list)

  idxs = set()
  with open(f'{results_dir}/{results_dataset}.json', 'r') as f:
      data = json.load(f)
  for d in data:
      example_to_choices[d['example_idx']].append((d['log_likelihood'], d['score']))
      example_to_choices_pmi_dc[d['example_idx']].append((d['log_likelihood'] - d['log_likelihood_given_domain_prefix'], d['score']))
      idx_to_correct_target_perplexities[language][d['example_idx']] = np.log(d['perplexity_of_correct_choice'])
      idx_to_input_perplexities[language][d['example_idx']] = np.log(d['perplexity_of_input'])
      idx_to_correct_target_probabilities[language][d['example_idx']] = np.exp(d['log_likelihood'])
      idxs.add(d['example_idx'])
  for example_idx, choices in example_to_choices.items():
      # accuracy by maximizing (non-length-normalized) log likelihood of choice
      if sorted(choices, reverse=True)[0][1] == 1:
          idx_to_correct[language][example_idx] = 1
      else:
          idx_to_correct[language][example_idx] = 0
  # accuracy by maximizing pmi_dc
  for example_idx, choices in example_to_choices_pmi_dc.items():
      if sorted(choices, reverse=True)[0][1] == 1:
          idx_to_correct_pmi_dc[language][example_idx] = 1
      else:
          idx_to_correct_pmi_dc[language][example_idx] = 0
  idxs = list(idxs)
  d = {'Dataset': dataset, 'Language': language, 'Benchmark suite': 'Stackexchange'}
  correct = list(idx_to_correct[language].values())
  correct_pmi_dc = list(idx_to_correct_pmi_dc[language].values())
  correct_target_perplexities = list(idx_to_correct_target_perplexities[language].values())
  input_perplexities = list(idx_to_input_perplexities[language].values())
  accuracy = sum(correct)/len(correct)
  accuracy_pmi_dc = sum(correct_pmi_dc)/len(correct_pmi_dc)
  d['Accuracy (LM)'] = accuracy
  d['Accuracy (PMI_DC)'] = accuracy_pmi_dc
  d['Normalized score (LM)'] = normalize_score(accuracy)
  d['Log perplexity of correct target'] = np.mean(correct_target_perplexities)
  d['Log perplexity of input'] = np.mean(input_perplexities)
  print(f'{dataset}: {accuracy} / {accuracy_pmi_dc}')

data_full_languages = []
english_not_in_yet = True
all_data = []
for other_language in languages:
    percentages = [0, 100]
    print(other_language)
    prev_num_to_translate = 0
    for percentage in percentages:
        d = {'Dataset': base_dataset, 'Language': other_language, 'Percentage': percentage, 'Benchmark suite': 'StackExchange'}
        num_to_translate = int(percentage/100 * len(idxs))
        english_idxs = idxs[num_to_translate:]
        other_idxs = idxs[:num_to_translate]
        correct = []
        correct_pmi_dc = []
        correct_target_perplexities = []
        input_perplexities = []
        pythia_dataset_total_counts = np.zeros(vocab_size)

        for language_idxs, language in zip([english_idxs, other_idxs],
                                        ['english', other_language]):
            correct.extend([idx_to_correct[language][idx] for idx in language_idxs])
            correct_pmi_dc.extend([idx_to_correct_pmi_dc[language][idx] for idx in language_idxs])
            correct_target_perplexities.extend([idx_to_correct_target_perplexities[language][idx] for idx in language_idxs])
            input_perplexities.extend([idx_to_input_perplexities[language][idx] for idx in language_idxs])
            # construct this mixed dataset's token distribution
            for idx in language_idxs:
                pythia_dataset_total_counts += idx_to_pythia_unigram_counts[language][idx]
            
        
        accuracy = sum(correct)/len(correct)
        accuracy_pmi_dc = sum(correct_pmi_dc)/len(correct_pmi_dc)

        
        # 1. compare token distribution to the pretraining distribution
        dataset_dist = smooth(normalize_dist(pythia_dataset_total_counts))    
        kls = []
        for pretraining_dist_pythia in pretraining_dists_pythia:
            kls.append(np.sum(rel_entr(dataset_dist, pretraining_dist_pythia)))
        res = bootstrap((kls,), np.mean)
        print(res)
        d['KL-divergences from the Pile (Pythia unigrams)'] = kls
        d['KL-divergence from the Pile (Pythia unigrams)'] = np.mean(kls)
        d['KL-divergence from the Pile (Pythia unigrams) (low CI)'] = res.confidence_interval.low
        d['KL-divergence from the Pile (Pythia unigrams) (high CI)'] = res.confidence_interval.high

        d['Accuracy (LM)'] = accuracy
        d['Normalized score (LM)'] = normalize_score(accuracy)
        d['Accuracy (PMI_DC)'] = accuracy_pmi_dc
        d['Normalized score (PMI_DC)'] = normalize_score(accuracy_pmi_dc)
        d['Log perplexity of correct target'] = np.mean(correct_target_perplexities)
        d['Log perplexity of input'] = np.mean(input_perplexities)
        all_data.append(d)

        if percentage == 0 and english_not_in_yet:
            e = d.copy()
            e['Language'] = 'english'
            data_full_languages.append(e)
            english_not_in_yet = False
        elif percentage == 100:
            data_full_languages.append(d)

        print(f'{percentage}: {accuracy} / {accuracy_pmi_dc}')
   
df = pd.DataFrame(all_data)
df_full_languages = pd.DataFrame(data_full_languages)

df.to_csv(f'{results_dir}/df_all_data.csv')
df_full_languages.to_csv(f'{results_dir}/df_full_languages.csv')

languages = ['english'] + languages
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
for _, row in df_full_languages.iterrows():
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
savefig(f'{output_dir}/figure_stackexchange-ampm_kl-divergence-pythia-unigrams_vs_normalized-score-lm.pdf', bbox_inches='tight')
plt.close()