import json
from collections import defaultdict
import numpy as np
import random
import pandas as pd
import os
from tqdm import tqdm
import operator
from scipy.special import rel_entr
from scipy.stats import bootstrap


def normalize_score(acc):
    low = 1/3
    return 100 * (acc - low) / (1 - low)

eps = 1e-8
num_buckets = 10000
vocab_size = 51000

def hash_buckets(string, num_buckets):
    return int(abs(hash(string)) % num_buckets)

def hash_counts(ngram_to_counts):
    hashed_counts = np.zeros(num_buckets, dtype=np.float32)
    for ngram, count in tqdm(ngram_to_counts.items()):
        hashed_counts[hash_buckets(ngram, num_buckets=num_buckets)] += count
    return hashed_counts

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

# load pretraining tokens and construct hashed distribution
print('Hashing pretraining WPT token distribution.')
with open('./n-gram-frequencies/the_pile-train_100000-docs_sample-1_wpt-counts-unigram-bigram-tokens_unbucketed.json', 'r') as f:
    pretraining_ngram_to_counts = json.load(f)
pretraining_dist_wpt = smooth(normalize_dist(hash_counts(pretraining_ngram_to_counts)))

print('Getting pretraining Pythia token distributions.')
num_pile_samples = 8
pretraining_dists_pythia = []
for sample_num in range(1, num_pile_samples + 1):
    with open(f'./n-gram-frequencies/the_pile-train_100000-docs_sample-{sample_num}_frequencies-unigram-tokens.json', 'r') as f:
        pretraining_ngram_to_counts = json.load(f)
    pretraining_dists_pythia.append(smooth(normalize_dist(vectorify(pretraining_ngram_to_counts))))

data_dir = './data-xnli'
results_dir = './xnli-output-pythia-6.9b'
output_dir = f'{results_dir}/graphs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def get_only_idxs(A, idxs):
    return [A[idx] for idx in idxs]

idx_to_correct = defaultdict(dict)
idx_to_correct_target_perplexities = defaultdict(dict)
idx_to_correct_target_probabilities = defaultdict(dict)
idx_to_input_perplexities = defaultdict(dict)

idx_to_wpt_ngram_counts_hashed = defaultdict(dict)
idx_to_pythia_unigram_counts = defaultdict(dict)

langauge_to_max_sims_in_order = {}
language_to_embeddings_in_order = {}

data_with_percents = []
languages = ['french', 'swahili', 'spanish', 'german',
             'bulgarian', 'russian', 'turkish', 'arabic', 'vietnamese',
             'greek', 'thai', 'chinese', 'hindi', 'urdu']
for percent in [25, 50, 75, 100]:
  for language in ['english'] + languages:
    if language == 'english' and percent != 100:
      continue
    print(language)
    dataset = f'xnli_{percent}-percent-{language}'
    d = {'Dataset': dataset, 'Language': language, 'Benchmark suite': 'XNLI'}

    # first load token distributions
    # WPT (Xie et al.)
    with open(f'{data_dir}/{dataset}-prompted_wpt-idx-to-counts-unigram-bigram-tokens_unbucketed.json', 'r') as f:
      idx_to_ngram_counts = json.load(f)
    for idx, ngram_to_counts in idx_to_ngram_counts.items():
      idx_to_wpt_ngram_counts_hashed[language][idx] = hash_counts(ngram_to_counts)
    # Pythia tokens
    with open(f'{data_dir}/{dataset}-prompted_pythia-2.8b-idx-to-counts-unigram-tokens.json', 'r') as f:
      idx_to_counts = json.load(f)
    pythia_dataset_total_counts = np.zeros(vocab_size)
    for idx, token_to_counts in idx_to_counts.items():
      counts = vectorify(token_to_counts)
      pythia_dataset_total_counts += counts
      idx_to_pythia_unigram_counts[language][idx] = counts

    # 1. compare token distribution to the pretraining distribution
    dataset_dist = smooth(normalize_dist(pythia_dataset_total_counts))    
    kls = []
    for pretraining_dist_pythia in pretraining_dists_pythia:
        kls.append(np.sum(rel_entr(dataset_dist, pretraining_dist_pythia)))
    res = bootstrap((kls,), np.mean)
    d['KL-divergences from the Pile (Pythia unigrams)'] = kls
    d['KL-divergence from the Pile (Pythia unigrams)'] = np.mean(kls)
    d['KL-divergence from the Pile (Pythia unigrams) (low CI)'] = res.confidence_interval.low
    d['KL-divergence from the Pile (Pythia unigrams) (high CI)'] = res.confidence_interval.high

    example_to_choices = defaultdict(list)

    idxs = set()
    with open(f'{results_dir}/{dataset}-validation.json', 'r') as f:
        data = json.load(f)
    for e in data:
        example_to_choices[e['in_order_example_idx']].append((e['ll'], e['score']))
        idx_to_correct_target_perplexities[language][e['in_order_example_idx']] = np.log(e['perplexity_of_target'])
        idx_to_correct_target_probabilities[language][e['in_order_example_idx']] = np.exp(e['ll'])
        idxs.add(e['in_order_example_idx'])
    for example_idx, choices in example_to_choices.items():
        # accuracy by maximizing (non-length-normalized) log likelihood of choice
        if sorted(choices, reverse=True)[0][1] == 1:
            idx_to_correct[language][example_idx] = 1
        else:
            idx_to_correct[language][example_idx] = 0
    idxs = list(idxs)
    correct = list(idx_to_correct[language].values())
    correct_target_perplexities = list(idx_to_correct_target_perplexities[language].values())
    accuracy = sum(correct)/len(correct)
    d['Accuracy (LM)'] = accuracy
    d['Normalized score (LM)'] = normalize_score(accuracy)
    d['Log perplexity of correct target'] = np.mean(correct_target_perplexities)
    d['Percentage'] = percent
    print(f'{dataset}: {accuracy}')
    data_with_percents.append(d)
    # add copies of english to represent 0% of other languages
    if language == 'english' and percent == 100:
        for other_language in languages:
            d = d.copy()
            d['Language'] = other_language
            d['Percentage'] = 0
            data_with_percents.append(d)

data_with_percents.sort(key=lambda d: d['Percentage'])

df = pd.DataFrame(data_with_percents)
df.to_csv(f'{results_dir}/df_percentages_partial.csv')
