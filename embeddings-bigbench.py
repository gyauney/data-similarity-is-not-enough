import random
import json
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

print('Initializing model.')
device = 'cuda'
model = SentenceTransformer('sentence-transformers/sentence-t5-base').to(device)

bb_datasets =  ['hindu_knowledge', 'bbq_lite_json', 'code_line_description', 'conceptual_combinations', 'emoji_movie', 'formal_fallacies_syllogisms_negation', 'known_unknowns', 'language_identification',
                'logic_grid_puzzle', 'logical_deduction', 'misconceptions_russian', 'novel_concepts', 'play_dialog_same_or_different',
                'strange_stories', 'strategyqa', 'symbol_interpretation', 'vitaminc_fact_verification', 'winowhy', 'auto_debugging',
                'conlang_translation', 'linguistics_puzzles', 'operators', 'parsinlu_reading_comprehension', 'repeat_copy_logic']

for bb_dataset in bb_datasets:
    dataset = load_dataset("bigbench", bb_dataset, split='default')
    file_prefix = f'bigbench_{bb_dataset}_default-split_all-docs'
    num_documents = len(dataset)
    print(f'{bb_dataset}: {num_documents} documents')
    print('Getting embeddings.')
    embeddings = model.encode(dataset['inputs'])
    # save embeddings!
    print('Saving embeddings.')
    np.save(f'./data/{file_prefix}_embeddings.npy', embeddings)
    print(f'Embedding shape: {embeddings.shape}')
    print()
