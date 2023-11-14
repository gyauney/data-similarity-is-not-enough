import json
from collections import defaultdict
import numpy as np
import pandas as pd
import os
import random
from tqdm import tqdm
# don't let matplotlib use xwindows
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pylab import savefig
import seaborn as sns
sns.set_style("whitegrid")
from scipy.special import rel_entr
from scipy.stats import pearsonr
from scipy.stats import bootstrap
import mauve

base_dataset = 'bigbench-lite-mul'

eps = 1e-8
num_buckets = 10000
vocab_size = 51000

# load embeddings
pretraining_embeddings = np.load('./data/the_pile-train_100000-docs_sample-1_embeddings.npy')
small_pretraining_embeddings = pretraining_embeddings[:1000, :]
def max_cosine_sims(pretraining_embeddings, dataset_embeddings):
    return np.max(dataset_embeddings @ pretraining_embeddings.T, axis=1)

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


def get_only_idxs(A, idxs):
    return [A[idx] for idx in idxs]


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


data_dir = './bigbench-datasets'
model_filename = 'pythia-6.9b'
results_dir = f'bigbench-output-{model_filename}'
output_dir = f'{results_dir}/graphs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

bb_mul_datasets =  ['hindu_knowledge', 'bbq_lite_json', 'code_line_description', 'conceptual_combinations', 'emoji_movie', 'formal_fallacies_syllogisms_negation', 'known_unknowns',
                    'language_identification',
                    'logic_grid_puzzle', 'logical_deduction',
                    #'misconceptions_russian',
                    'novel_concepts', 'play_dialog_same_or_different',
                    'strange_stories', 'strategyqa', 'symbol_interpretation', 'vitaminc_fact_verification', 'winowhy']

dataset_to_random_guess_accuracy = {
    "hindu_knowledge": 0.2480952380952381,
    "bbq_lite_json": 0.3333333333333082,
    "code_line_description": 0.24833333333333335,
    "conceptual_combinations": 0.25,
    "emoji_movie": 0.19999999999999962,
    "formal_fallacies_syllogisms_negation": 0.5,
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

dataset_to_data = {d: {'Dataset': d, 'Benchmark suite': 'BIG-bench Lite'} for d in bb_mul_datasets}

for bb_dataset in bb_mul_datasets:
    example_to_choices = defaultdict(list)
    correct = []
    correct_target_perplexities = []
    correct_target_probabilities = []
    with open(f'{results_dir}/{bb_dataset}.json', 'r') as f:
        data = json.load(f)
    for d in data:
        example_to_choices[d['example_idx']].append((d['ll'], d['score']))
        correct_target_perplexities.append(d['perp'])
        correct_target_probabilities.append(np.exp(d['ll']))
    for example_idx, choices in example_to_choices.items():
        # accuracy by maximizing (non-length-normalized) log likelihood of choice
        if sorted(choices, reverse=True)[0][1] == 1:
            correct.append(1)
        else:
            correct.append(0)
    accuracy = sum(correct)/len(correct)
    
    # first load token distributions
    # WPT (Xie et al.)
    with open(f'{data_dir}/{bb_dataset}_wpt-counts-unigram-bigram-tokens_unbucketed.json', 'r') as f:
        ngram_to_counts = json.load(f)
    dataset_dist = smooth(normalize_dist(hash_counts(ngram_to_counts)))
    dataset_to_data[bb_dataset]['KL-divergence from the Pile (Hashed WPT bigrams)'] = np.sum(rel_entr(dataset_dist, pretraining_dist_wpt))

    # Pythia tokens
    with open(f'{data_dir}/{bb_dataset}_pythia-2.8b-counts-unigram-tokens.json', 'r') as f:
        token_to_counts = json.load(f)
    dataset_dist = smooth(normalize_dist(vectorify(token_to_counts)))
    kls = []
    for pretraining_dist_pythia in pretraining_dists_pythia:
        kls.append(np.sum(rel_entr(dataset_dist, pretraining_dist_pythia)))
    res = bootstrap((kls,), np.mean)
    dataset_to_data[bb_dataset]['KL-divergences from the Pile (Pythia unigrams)'] = kls
    dataset_to_data[bb_dataset]['KL-divergence from the Pile (Pythia unigrams)'] = np.mean(kls)
    dataset_to_data[bb_dataset]['KL-divergence from the Pile (Pythia unigrams) (low CI)'] = res.confidence_interval.low
    dataset_to_data[bb_dataset]['KL-divergence from the Pile (Pythia unigrams) (high CI)'] = res.confidence_interval.high
        
    # now get mauve score
    dataset_embeddings = np.load(f'./data/bigbench_{bb_dataset}_default-split_all-docs.npy')
    out = mauve.compute_mauve(p_features=small_pretraining_embeddings, q_features=dataset_embeddings, num_buckets=50)
    dataset_to_data[bb_dataset]['MAUVE score'] = out.mauve

    dataset_to_data[bb_dataset]['Accuracy (LM)'] = accuracy
    dataset_to_data[bb_dataset]['Normalized score (LM)'] = normalize_score(bb_dataset, accuracy)
    dataset_to_data[bb_dataset]['Perplexity of correct target'] = np.median(correct_target_perplexities)
    dataset_to_data[bb_dataset]['Probability of correct target'] = np.median(correct_target_probabilities)
    print(f'{bb_dataset}: {accuracy:.8f} max ll')

df = pd.DataFrame(dataset_to_data.values())
df.to_csv(f'{results_dir}/df_bigbench_with_similarities.csv')
