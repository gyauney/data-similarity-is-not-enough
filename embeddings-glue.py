import random
import json
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

print('Initializing model.')
device = 'cuda'
model = SentenceTransformer('sentence-transformers/sentence-t5-base').to(device)

datasets =  [('mnli', 'validation_matched'),
            ('mnli', 'validation_mismatched'),
            ('qnli', 'validation'),
            ('qqp', 'validation'),
            ('sst2', 'validation'),
            ('rte', 'validation'),
            ]

for dataset_name, split in datasets:

    with open(f'./data/glue_{dataset_name}_{split}-prompted.json', 'r') as f:
        examples = json.load(f)
    texts = [e['text'] for e in examples]
    file_prefix = f'glue_{dataset_name}_{split}'
    num_documents = len(texts)
    print(f'{dataset_name}_{split}: {num_documents} documents')
    print('Getting embeddings.')
    embeddings = model.encode(texts)
    # save embeddings!
    print('Saving embeddings.')
    np.save(f'./data/{file_prefix}_embeddings.npy', embeddings)
    print(f'Embedding shape: {embeddings.shape}')
    print()
