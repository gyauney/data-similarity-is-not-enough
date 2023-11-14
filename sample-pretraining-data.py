import random
import json
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--sample_num', required=True, type=int)
parser.add_argument('--num_samples', required=True, type=int)
parser.add_argument('--dataset_name', required=True, type=str)
args = parser.parse_args()

if not os.path.exists('./data'):
    os.makedirs('./data')

num_samples = args.num_samples
if args.dataset_name == 'c4':
    pretraining_name = 'c4'
    pretraining_filename = 'c4'
    split = 'train'
    config = 'en'
elif args.dataset_name == 'the_pile':
    pretraining_name = 'EleutherAI/pile'
    pretraining_filename = 'the_pile'
    split = 'train'
    config = 'all'
else:
    print(f'Unknown dataset_name: {args.dataset_name}')
    exit()

file_prefix = f'{pretraining_filename}-{split}_{num_samples}-docs_sample-{args.sample_num}'

dataset = load_dataset(pretraining_name, config,
                       split=split)
num_documents = len(dataset)

print(dataset[0])
print(f'Number of documents: {num_documents}')

# sample 100,000 random idxs
sampled_idxs = random.sample(range(num_documents), num_samples)
with open(f'./data/{file_prefix}_idxs.json', 'w') as f:
    json.dump(sampled_idxs, f)
assert(len(set(sampled_idxs)) == num_samples)
# The Pile doesn't have keys...
if dataset == 'c4':
    keys = [dataset[idx]['url'] for idx in sampled_idxs]
    with open(f'./data/{file_prefix}_urls.json', 'w') as f:
        json.dump(keys, f)
texts = [dataset[idx]['text'] for idx in sampled_idxs]
with open(f'./data/{file_prefix}_texts.json', 'w') as f:
    json.dump(texts, f)

# get embeddings!!
print('Initializing model.')
device = 'cuda'
model = SentenceTransformer('sentence-transformers/sentence-t5-base').to(device)
sentences = [dataset[idx]['text'] for idx in sampled_idxs]
print('Getting embeddings.')
embeddings = model.encode(sentences)

# save embeddings!
print('Saving embeddings.')
np.save(f'./data/{file_prefix}_embeddings.npy', embeddings)

print(f'Embedding shape: {embeddings.shape}')
