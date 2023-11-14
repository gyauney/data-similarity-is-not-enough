import numpy as np
import argparse
from collections import defaultdict
from datasets import Dataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch
import json
import os
import random
import time
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, type=str)
parser.add_argument('--label_type', required=True, type=str)
args = parser.parse_args()

base_dataset = 'bicycles-cstheory'
label_type = args.label_type
if label_type == 'forum':
  prompt_question = 'This post is about'
  possible_choices = ['bicycles', 'cstheory']
elif label_type == 'ampm':
  prompt_question = 'This was posted in the'
  possible_choices = ['morning', 'afternoon']

device = "cuda"
model_string = args.model
model_filename_string = model_string.lower()
model_id = f"EleutherAI/{model_filename_string}"
model = GPTNeoXForCausalLM.from_pretrained(model_id).to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

data_dir = './data-staxckexchange'
output_dir = f'./stackexchange-output-{model_filename_string}_{label_type}-labels'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

dataset_to_avg_random_guess = {}

def hour_to_ampm(hour):
  if hour < 12:
    return 'morning'
  else:
    return 'afternoon'

# load and tokenize the dataset
def get_multiple_choice_dataset(dataset_name):
  
  with open(f'{data_dir}/{dataset_name}.json', 'r') as f:
    examples = json.load(f)

  df = pd.DataFrame(examples)
  if label_type == 'forum':
    dataset = Dataset.from_dict({'input_pretokenized': df['text'].tolist(), 'answer_pretokenized': df['forum'].tolist(), 'language': df['language'].tolist(), 'idx': df['id'].tolist()})
  elif label_type == 'ampm':
    dataset = Dataset.from_dict({'input_pretokenized': df['text'].tolist(), 'answer_pretokenized': [hour_to_ampm(h) for h in df['hour_posted']], 'language': df['language'].tolist(), 'idx': df['id'].tolist()})

  #dataset = dataset.add_column("idx", list(range(len(dataset))))

  def templatize_examples(example):
    example['input_pretokenized'] = f"{example['input_pretokenized']}\n{prompt_question}"
    return example
  def concatenate_choices(example):
    batch = defaultdict(list)
    all_keys = set(example.keys())
    for i, choice in enumerate(possible_choices):
      batch['idx'].append(f'{example["idx"][0]}_{i}')
      batch['example_idx'].append(example["idx"][0])
      batch['answer_idx'].append(i)
      batch['inputs'].append(f'{example["input_pretokenized"][0]} {choice}')
      batch['inputs_without_choice'].append(example["input_pretokenized"][0])
      batch['domain_prefix_choice'].append(f'{example["domain_prefix"][0]}{choice}')
      batch['score'].append(int(choice == example['answer_pretokenized'][0]))
      batch['answer_pretokenized'].append(example['answer_pretokenized'][0])
      batch['choice'].append(choice)
      # fill up the other fields with placeholders of the correct type
      for k in all_keys - set(['idx', 'answer_pretokenized']):
        batch[k].append(type(example[k][0])())
    return batch
  def mask_question_tokens(example):
    example['target_ids'] = np.array(example['input_ids'].copy())
    choice_tokens = tokenizer(example["choice"], return_tensors="np", max_length=512, padding=False, truncation=True)
    num_choice_tokens = choice_tokens.input_ids.shape[1]
    zero_idxs = np.where(np.array(example['attention_mask']) == 0)[0]
    if len(zero_idxs) == 0:
      last_idx = len(example['input_ids'])
    else:
      last_idx = zero_idxs[0]
    example['num_choice_tokens'] = num_choice_tokens
    example['target_ids'][:last_idx-num_choice_tokens] = -100
    example['target_ids'][last_idx:] = -100
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
  dataset = dataset.map(templatize_examples, batched=False)
  dataset = dataset.add_column("domain_prefix", [f'{prompt_question} '] * len(dataset))
  dataset = dataset.map(concatenate_choices, batched=True, batch_size=1)
  dataset = dataset.map(lambda examples: tokenizer(examples["inputs"], return_tensors="np", max_length=512, padding=True, truncation=True), batched=True, batch_size=1)
  dataset = dataset.map(tokenize_inputs_without_choice, batched=False)
  dataset = dataset.map(tokenize_domain_prefix, batched=False)
  dataset = dataset.map(mask_question_tokens, batched=False)
  dataset = dataset.map(mask_domain_prefix_tokens, batched=False) 

  # save prompted texts for easy similarity calculation
  to_save = [{'text': example['inputs'], 'id': example['example_idx']} for example in dataset if example['score']]
  with open(f'{data_dir}/{dataset_name}_{label_type}-labels-prompted.json', 'w') as f:
    json.dump(to_save, f, ensure_ascii=False)
  print(f'Saving {len(to_save)} docs')
  return dataset

times = {}

for percent in [25, 50, 75, 100]:
    for language in [
                    'finnish', 'spanish', 'french',
                    'german',
                    'indonesian',
                    'swahili'
                    ]:
        dataset_name = f'{base_dataset}_{percent}-percent-{language}'
        print(dataset_name)
        
        dataset = get_multiple_choice_dataset(dataset_name)

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
                    'language': example['language'],
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
                    }
            results.append(result)
            del input_ids
            del attention_mask
            del target_ids
        end = time.time()
        times[dataset_name] = end - start
        with open(f'{output_dir}/{dataset_name}.json', 'w') as f:
            json.dump(results, f)
        torch.cuda.empty_cache()

with open(f'{output_dir}/times.json', 'w') as f:
    json.dump(times, f)