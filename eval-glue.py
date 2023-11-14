import numpy as np
import argparse
from collections import defaultdict
from datasets import load_dataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch
import json
import os
import random
import time
from promptsource.templates import DatasetTemplates

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, type=str)
args = parser.parse_args()

device = "cuda"
model_string = args.model
model_filename_string = model_string.lower()
model_id = f"EleutherAI/{model_filename_string}"
model = GPTNeoXForCausalLM.from_pretrained(model_id).to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token


output_dir = f'./glue-output-{model_filename_string}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print('loaded the model!')

dataset_to_avg_random_guess = {}

# load and tokenize the dataset
def get_multiple_choice_dataset_prompt_variant(dataset_name, split, subsample, num_to_subsample=None):
  dataset = load_dataset("glue", dataset_name, split=split)
  
  possible_choices = dataset_to_possible_choices[dataset_name]

  # First subsample the dataset
  if subsample and num_to_subsample < len(dataset):
    print(f'{dataset_name}: sampling {num_to_subsample} out of {len(dataset)} examples.')
    subsampled_idxs_filename = f'./glue-subsampled-idxs/{dataset_name}_{split}_{num_to_subsample}-examples_idxs.json'
    if os.path.exists(subsampled_idxs_filename):
      with open(subsampled_idxs_filename, 'r') as f:
        idxs_to_subsample = json.load(f)
    else:
      idxs_to_subsample = random.sample(range(len(dataset)), num_to_subsample)
      with open(subsampled_idxs_filename, 'w') as f:
        json.dump(idxs_to_subsample, f)
    dataset = dataset.select(idxs_to_subsample)
  def concatenate_choices(example):
    batch = defaultdict(list)
    all_keys = set(example.keys())
    for i, choice in enumerate(possible_choices):
      input_with_choice_pretokenized = prompt_fn(dataset_name, example, choice)
      answer_pretokenized = possible_choices[example["label"][0]]
      batch['idx'].append(f'{example["idx"][0]}_{i}')
      batch['example_idx'].append(example["idx"][0])
      batch['answer_idx'].append(i)
      batch['inputs'].append(input_with_choice_pretokenized)
      # can't do input without choice with this prompting setup
      batch['inputs_without_choice'].append(input_with_choice_pretokenized)
      batch['domain_prefix_choice'].append(f'{example["domain_prefix"][0]}{choice}')
      batch['score'].append(int(choice == answer_pretokenized))
      batch['answer_pretokenized'].append(answer_pretokenized)
      batch['choice'].append(choice)
      # fill up the other fields with placeholders of the correct type
      for k in all_keys - set(['idx', 'answer_pretokenized']):
        batch[k].append(type(example[k][0])())
    return batch
  def mask_question_tokens(example):
    example['target_ids'] = np.array(example['input_ids'].copy())
    example['num_choice_tokens'] = len(example['target_ids'])
    return example
  def tokenize_inputs_without_choice(example):
    tokenized = tokenizer(example['inputs_without_choice'], return_tensors="np", max_length=512, padding=True, truncation=True)
    example['inputs_without_choice_input_ids'] = tokenized.input_ids
    example['inputs_without_choice_attention_mask'] = tokenized.attention_mask
    return example
  def tokenize_domain_prefix(example):
    tokenized = tokenizer(example['domain_prefix_choice'], return_tensors="np", max_length=512, padding=True, truncation=True)
    example['domain_prefix_choice_input_ids'] = tokenized.input_ids
    example['domain_prefix_choice_attention_mask'] = tokenized.attention_mask
    return example
  def mask_domain_prefix_tokens(example):
    example['domain_prefix_choice_target_ids'] = np.array(example['domain_prefix_choice_input_ids'].copy())
    example['domain_prefix_choice_target_ids'][0, :-example['num_choice_tokens']] = -100
    return example
  dataset = dataset.add_column("domain_prefix", [dataset_to_domain_prefix[dataset_name]] * len(dataset))
  dataset = dataset.map(concatenate_choices, batched=True, batch_size=1)
  dataset = dataset.map(lambda examples: tokenizer(examples["inputs"], return_tensors="np", max_length=512, padding=True, truncation=True), batched=True, batch_size=1)
  dataset = dataset.map(tokenize_inputs_without_choice, batched=False)
  dataset = dataset.map(tokenize_domain_prefix, batched=False)
  dataset = dataset.map(mask_question_tokens, batched=False)
  dataset = dataset.map(mask_domain_prefix_tokens, batched=False)  

  to_save = [{'text': example['inputs'], 'id': example['example_idx']} for example in dataset if example['score']]
  with open(f'./data/glue_{dataset_name}_{split}-prompted.json', 'w') as f:
    json.dump(to_save, f)
  print(f'Saving {len(to_save)} docs')
  

  return dataset

datasets =  [('mnli', 'validation_matched'),
            ('mnli', 'validation_mismatched'),
            ('qnli', 'validation'),
            ('qqp', 'validation'),
            ('sst2', 'validation'),
            ('rte', 'validation'),
            ]
dataset_to_domain_prefix = {
  'mnli': ' ',
  'qnli': ' ',
  'qqp': ' ',
  'sst2': ' ',
  'rte': ' ',
} 

dataset_to_possible_choices = {
  'mnli': ['Yes', 'Maybe', 'No'],
  'qnli': ['Yes', 'No'],
  'qqp': ['No', 'Yes'],
  'sst2': ['negative', 'positive'],
  'rte': ['Yes', 'No'],
}

def prompt_fn(dataset, example, choice):
  if dataset == 'mnli':
    return f"{example['premise'][0]} {choice}, {example['hypothesis'][0]}"
  elif dataset == 'qnli':
    return f"{example['sentence'][0]} {example['question'][0]} {choice}"
  elif dataset == 'qqp':
    return f"Question 1: {example['question1'][0]}\nQuestion 2: {example['question2'][0]}\nDo these two questions convey the same meaning? {choice}"
  elif dataset == 'sst2':
    return f"Does the following sentence have a \"positive\" or \"negative\" sentiment?\n{example['sentence'][0]}\n{choice}"
  elif dataset == 'rte':
    return f"{example['sentence1'][0]} {choice}, {example['sentence2'][0]}"
  else:
    print(f'Dataset not implemented: {dataset}')
    exit()

times = {}

# all multiple choice tasks
for dataset_name, split in datasets:
  print(dataset_name, split)
  dataset = get_multiple_choice_dataset_prompt_variant(dataset_name, split=split, subsample=False)
  print(f'    {len(dataset)} documents.')
  start = time.time()
  results = []
  for i in tqdm(range(len(dataset))):
    example = dataset[i]
    if i < 2:
      print(f'Example {i}:')
      for k, v in example.items():
        print(f'{k}: {v}')
      print()
    input_ids = torch.tensor([example['input_ids']]).to(device)
    attention_mask = torch.tensor([example['attention_mask']]).to(device)
    target_ids = torch.tensor([example['target_ids']]).to(device)
    with torch.no_grad():
      outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)
    loss = outputs.loss.item()
    num_choice_tokens = example['num_choice_tokens']
    log_likelihood = -1 * loss * num_choice_tokens
    perplexity = np.exp(-1 * log_likelihood / num_choice_tokens)
    del input_ids
    del attention_mask
    del target_ids
    # now get the loss conditioned on just the domain prefix
    input_ids = torch.tensor(example['domain_prefix_choice_input_ids']).to(device)
    attention_mask = torch.tensor(example['domain_prefix_choice_attention_mask']).to(device)
    target_ids = torch.tensor(example['domain_prefix_choice_target_ids']).to(device)
    with torch.no_grad():
      outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)
    log_likelihood_given_domain_prefix = -1 * outputs.loss.item() * num_choice_tokens
    # and finally get the loss/perplexity of just the input with no choice
    input_ids = torch.tensor(example['inputs_without_choice_input_ids']).to(device)
    attention_mask = torch.tensor(example['inputs_without_choice_attention_mask']).to(device)
    num_input_tokens = input_ids.shape[1]
    with torch.no_grad():
      outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
    log_likelihood_input = -1 * outputs.loss.item() * num_input_tokens
    perplexity_input = np.exp(-1 * log_likelihood_input / num_input_tokens)
    result = {'dataset': dataset_name,
              'combined_idx': example['idx'],
              'example_idx': example['example_idx'],
              'answer_idx': example['answer_idx'],
              'score': example['score'],
              'num_choice_tokens': num_choice_tokens,
              'loss': loss,
              'log_likelihood': log_likelihood,
              'perplexity_of_correct_choice': perplexity,
              'log_likelihood_given_domain_prefix': log_likelihood_given_domain_prefix,
              'log_likelihood_input': log_likelihood_input,
              'perplexity_of_input': perplexity_input,
              'num_input_tokens': num_input_tokens,
              'inputs': example['inputs'],
              'answer': example['answer_pretokenized'],
              }
    results.append(result)
    del input_ids
    del attention_mask
    del target_ids
  end = time.time()
  times[dataset_name] = end - start
  with open(f'{output_dir}/{dataset_name}_{split}.json', 'w') as f:
    json.dump(results, f)
  torch.cuda.empty_cache()

with open(f'{output_dir}/times.json', 'w') as f:
    json.dump(times, f)