import json
from collections import defaultdict
import numpy as np
import random
import pandas as pd
import os
# don't let matplotlib use xwindows
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pylab import savefig
import seaborn as sns
sns.set_style("whitegrid")
import mauve
from scipy.stats import pearsonr
from scipy.stats import bootstrap


# load embeddings
pretraining_embeddings = np.load('./data/the_pile-train_100000-docs_sample-1_embeddings.npy')
small_pretraining_embeddings = pretraining_embeddings[:1000, :]
def max_cosine_sims(pretraining_embeddings, dataset_embeddings):
    return np.max(dataset_embeddings @ pretraining_embeddings.T, axis=1)

results_dir = './glue-output-pythia-6.9b'
output_dir = './glue-output-pythia-6.9b/graphs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

datasets = [('mnli', 'validation_matched'),
            ('mnli', 'validation_mismatched'),
            ('qnli', 'validation'),
            ('qqp', 'validation'),
            ('sst2', 'validation'),
            ('rte', 'validation'),
            ]

def get_only_idxs(A, idxs):
    return [A[idx] for idx in idxs]

all_data = []
for dataset, split in datasets:

    dataset = f'{dataset}_{split}'

    example_to_choices = defaultdict(list)
    example_to_choices_pmi_dc = defaultdict(list)
    
    idx_to_correct = {}
    idx_to_correct_pmi_dc = {}
    idx_to_correct_target_perplexities = {}
    idx_to_correct_target_probabilities = {}
    idx_to_input_perplexities = {}
    idxs = set()
    with open(f'{results_dir}/{dataset}.json', 'r') as f:
        data = json.load(f)
    for d in data:
        example_to_choices[d['example_idx']].append((d['log_likelihood'], d['score']))
        example_to_choices_pmi_dc[d['example_idx']].append((d['log_likelihood'] - d['log_likelihood_given_domain_prefix'], d['score']))
        idx_to_correct_target_perplexities[d['example_idx']] = np.log(d['perplexity_of_correct_choice'])
        idx_to_input_perplexities[d['example_idx']] = np.log(d['perplexity_of_input'])
        idx_to_correct_target_probabilities[d['example_idx']] = np.exp(d['log_likelihood'])
        idxs.add(d['example_idx'])
    for example_idx, choices in example_to_choices.items():
        # accuracy by maximizing (non-length-normalized) log likelihood of choice
        if sorted(choices, reverse=True)[0][1] == 1:
            idx_to_correct[example_idx] = 1
        else:
            idx_to_correct[example_idx] = 0
    # accuracy by maximizing pmi_dc
    for example_idx, choices in example_to_choices_pmi_dc.items():
        if sorted(choices, reverse=True)[0][1] == 1:
            idx_to_correct_pmi_dc[example_idx] = 1
        else:
            idx_to_correct_pmi_dc[example_idx] = 0
    idxs = list(idxs)
    
    correct = list(idx_to_correct.values())
    correct_pmi_dc = list(idx_to_correct_pmi_dc.values())
    correct_target_perplexities = list(idx_to_correct_target_perplexities.values())
    input_perplexities = list(idx_to_input_perplexities.values())
    
    accuracy = sum(correct)/len(correct)
    accuracy_pmi_dc = sum(correct_pmi_dc)/len(correct_pmi_dc)
    
    d = {'Dataset': dataset, 'Benchmark suite': 'GLUE'}
    d['Accuracy (LM)'] = accuracy
    d['Accuracy (PMI_DC)'] = accuracy_pmi_dc
    d['Log perplexity of correct target'] = np.mean(correct_target_perplexities)
    d['Log perplexity of input'] = np.mean(input_perplexities)
    all_data.append(d)

    dataset_embeddings = np.load(f'./data/glue_{dataset}-embeddings.npy')
    
    # load embeddings and get each document's max
    d['Max cosine similarities'] = max_cosine_sims(pretraining_embeddings, dataset_embeddings)
    out = mauve.compute_mauve(p_features=small_pretraining_embeddings, q_features=dataset_embeddings, num_buckets=50)
    d['MAUVE score'] = out.mauve

df = pd.DataFrame(all_data)
df.to_csv(f'{output_dir}/df_glue.csv')