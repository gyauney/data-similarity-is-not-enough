
'''
Implement the KL-divergence between hashed bigram distributions
from Xie et al., 2023.
'''
from itertools import islice
import os
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer
import argparse
from nltk.tokenize import WordPunctTokenizer


parser = argparse.ArgumentParser()
parser.add_argument('--sample_num', required=True, type=int)
parser.add_argument('--dataset_name', required=True, type=str)
args = parser.parse_args()

eps = 1e-8
num_buckets = 10000

wpt = WordPunctTokenizer()

model_string = 'Pythia-2.8B'
model_filename_string = model_string.lower()
model_id = f"EleutherAI/{model_filename_string}"
tokenizer = AutoTokenizer.from_pretrained(model_id)

results_dir = 'n-gram-frequencies'
pretraining_prefix = f'{args.dataset_name}-train_100000-docs_sample-{args.sample_num}'

if not os.path.exists(results_dir):
    os.makedirs(results_dir)


print('Tokenizing pretraining texts')
with open(f'./data/{pretraining_prefix}_texts.json', 'r') as f:
    pretraining_texts = json.load(f)
all_tokenizations = []
for text in tqdm(pretraining_texts):
    tokenized = tokenizer(text, return_tensors="np", truncation=False)
    tokens = tokenized.input_ids[0, :]
    tokens = list([int(t) for t in tokens])
    all_tokenizations.append(tokens)

# start with Pythia unigrams
frequencies = defaultdict(int)
for tokens in tqdm(all_tokenizations):
    for token in tokens:
        frequencies[token] += 1

with open(f'{results_dir}/{pretraining_prefix}_frequencies-unigram-tokens.json', 'w') as f:
    json.dump(frequencies, f, indent=2)

# for hashing later
print('First just count uni-/bigrams in order to hash later.')
counts = defaultdict(int)
for text in tqdm(pretraining_texts):
    words = wpt.tokenize(text.lower())
    unigrams, bigrams = words, list(zip(words, islice(words, 1, None)))
    for unigram in unigrams:
        counts[str(unigram)] += 1
    for bigram in bigrams:
        counts[' '.join([str(t) for t in bigram])] += 1
with open(f'{results_dir}/{pretraining_prefix}_wpt-counts-unigram-bigram-tokens_unbucketed.json', 'w') as f:
    json.dump(counts, f, indent=2, ensure_ascii=False)
print(f'{results_dir}/{pretraining_prefix}_wpt-counts-unigram-bigram-tokens_unbucketed.json')