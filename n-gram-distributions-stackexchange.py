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

data_dir = 'data-stackexchange'
base_dataset = 'bicycles-cstheory'

for percent in [25, 50, 75, 100]:
    for language in ['english', 'finnish', 'spanish', 'french', 'german', 'indonesian', 'swahili']:
        if language == 'english' and percent != 100:
            continue
        print(language)
        dataset_name = f'{base_dataset}_{percent}-percent-{language}_forum-labels-prompted'
        with open(f'./{data_dir}/{dataset_name}.json', 'r') as f:
            examples = json.load(f)
        print(f'Tokenizing {dataset_name}')
        all_tokenizations = []
        texts = [e['text'] for e in examples]
        idxs = [e['id'] for e in examples]
        for text in tqdm(texts):
            tokenized = tokenizer(text, return_tensors="np", truncation=False)
            tokens = tokenized.input_ids[0, :]
            tokens = list([int(t) for t in tokens])
            all_tokenizations.append(tokens)
        
        print(f'{language}: counting unigram tokens')
        idx_to_counts = {}
        for idx, tokens in tqdm(zip(idxs, all_tokenizations)):
            counts = defaultdict(int)
            for token in tokens:
                counts[str(token)] += 1
            idx_to_counts[idx] = counts
        with open(f'{data_dir}/{dataset_name}_{model_filename_string}-idx-to-counts-unigram-tokens.json', 'w') as f:
            json.dump(idx_to_counts, f, indent=2)