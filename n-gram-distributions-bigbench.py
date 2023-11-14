import os
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd
from googletrans import Translator
import random
import time
import copy
from itertools import islice
import string
from nltk.tokenize import WordPunctTokenizer

wpt = WordPunctTokenizer()

model_string = 'Pythia-2.8B'
model_filename_string = model_string.lower()
model_id = f"EleutherAI/{model_filename_string}"
tokenizer = AutoTokenizer.from_pretrained(model_id)

data_dir = './bigbench-datasets'

bb_mul_datasets =  ['hindu_knowledge', 'bbq_lite_json', 'code_line_description', 'conceptual_combinations', 'emoji_movie', 'formal_fallacies_syllogisms_negation', 'known_unknowns', 'language_identification',
                    'logic_grid_puzzle', 'logical_deduction', 'misconceptions_russian', 'novel_concepts', 'play_dialog_same_or_different',
                    'strange_stories', 'strategyqa', 'symbol_interpretation', 'vitaminc_fact_verification', 'winowhy']
bb_gen_datasets = ['auto_debugging',
                   'conlang_translation', 'linguistics_puzzles', 'operators', 'parsinlu_reading_comprehension', 'repeat_copy_logic']

for dataset_name in bb_mul_datasets + bb_gen_datasets:
    print(dataset_name)
    # with open(f'{data_dir}/{dataset_name}.json', 'r') as f:
    #     examples = json.load(f)
    
    dataset = load_dataset("bigbench", dataset_name, split='default')
    idxs = dataset['idx']
    texts = dataset['inputs']

    print(f'Tokenizing {dataset_name}')
    all_tokenizations = []
    # texts = [e['text'] for e in examples]
    # idxs = [e['id'] for e in examples]
    for text in tqdm(texts):
        tokenized = tokenizer(text, return_tensors="np", truncation=False)
        tokens = tokenized.input_ids[0, :]
        tokens = list([int(t) for t in tokens])
        all_tokenizations.append(tokens)

    print(f'{dataset_name}: counting unigram tokens')
    counts = defaultdict(int)
    for idx, tokens in tqdm(zip(idxs, all_tokenizations)):
        for token in tokens:
            counts[str(token)] += 1
    with open(f'{data_dir}/{dataset_name}_{model_filename_string}-counts-unigram-tokens.json', 'w') as f:
        json.dump(counts, f, indent=2)

    # bin!!!
    print(f'{dataset_name}: counting unigram + bigram WPT tokens for Xie et al. baseline')
    counts = defaultdict(int)
    for idx, text in tqdm(zip(idxs, texts)):
        words = wpt.tokenize(text.lower())
        unigrams, bigrams = words, list(zip(words, islice(words, 1, None)))
        for unigram in unigrams:
            counts[str(unigram)] += 1
        for bigram in bigrams:
            counts[' '.join([str(t) for t in bigram])] += 1
    with open(f'{data_dir}/{dataset_name}_wpt-counts-unigram-bigram-tokens_unbucketed.json', 'w') as f:
        json.dump(counts, f, indent=2, ensure_ascii=False)
