import numpy as np
import pandas as pd
import argparse
from collections import defaultdict
from datasets import load_dataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch
import json
import os
from datasets import Dataset

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

output_dir = f'./xnli-output-{model_filename_string}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

yes_maybe_no = ['Yes', 'Maybe', 'No']

# load and tokenize the dataset
def get_multiple_choice_dataset_prompt_variant(dataset_name):
  with open(f'./data-xnli/{dataset_name}.json', 'r') as f:
    examples = json.load(f)
  df = pd.DataFrame(examples)
  dataset = Dataset.from_dict({'premise': df['premise'].tolist(), 'hypothesis': df['hypothesis'].tolist(), 'label': df['label'].tolist(), 'in_order_example_idx': df['id'].tolist()})
  # create new examples, each with a different target choice
  # example is a batch--so all its fields are lists of length 1
  def concatenate_choices(example):
    batch = defaultdict(list)
    choices = yes_maybe_no
    for i, choice in enumerate(choices):
      batch['combined_idx'].append(f'{example["in_order_example_idx"][0]}_{i}')
      batch['in_order_example_idx'].append(example["in_order_example_idx"][0])
      batch['answer_idx'].append(i)
      batch['inputs'].append(f"{example['premise'][0]}? {choice}, {example['hypothesis'][0]}")
      batch['score'].append(i == example['label'][0])
      batch['choice'].append(choice)
      batch['premise'].append(example['premise'][0])
      batch['hypothesis'].append(example['hypothesis'][0])
      batch['label'].append(example['label'][0])
    return batch
  # n.b. getting loss of the ENTIRE example, not just the answer choice
  def mask_question_tokens(example):
    example['target_ids'] = np.array(example['input_ids'].copy())
    example['num_choice_tokens'] = len(example['target_ids'])
    return example
  dataset = dataset.map(concatenate_choices, batched=True, batch_size=1)
  dataset = dataset.map(lambda examples: tokenizer(examples["inputs"], return_tensors="np", max_length=512, padding=True, truncation=True), batched=True, batch_size=1)
  dataset = dataset.map(mask_question_tokens, batched=False)

  # save texts for easy similarity calculation
  to_save = [{'text': example['inputs'], 'id': example['in_order_example_idx']} for example in dataset if example['score']]
  with open(f'./data-xnli/{dataset_name}-prompted.json', 'w') as f:
    json.dump(to_save, f)
  print(f'Saving {len(to_save)} docs')

  return dataset

languages =  [('English', 'en'),
              ('Spanish', 'es'),
              ('German', 'de'),
              ('Swahili', 'sw'),
              ('French', 'fr'),
              ('Bulgarian', 'bg'),
              ('Russian', 'ru'),
              ('Turkish', 'tr'),
              ('Arabic', 'ar'),
              ('Vietnamese', 'vi'),
              ('Greek', 'el'),
              ('Thai', 'th'),
              ('Chinese', 'zh'),
              ('Hindi', 'hi'),
              ('Urdu', 'ur'),]

for percent in [25, 50, 75, 100]:
  for language_full, language in languages:
    if language_full == 'English' and percent != 100:
      continue
    print(language)
    dataset_name = f'xnli_{percent}-percent-{language_full.lower()}'
    dataset = get_multiple_choice_dataset_prompt_variant(language, dataset_name, percent)
    results = []
    for i in tqdm(range(len(dataset))):
      example = dataset[i]
      input_ids = torch.tensor([example['input_ids']]).to(device)
      attention_mask = torch.tensor([example['attention_mask']]).to(device)
      target_ids = torch.tensor([example['target_ids']]).to(device)
      with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)
      loss = outputs.loss.item()
      num_target_tokens = example['num_choice_tokens']
      log_likelihood = -1 * loss * num_target_tokens
      perplexity = np.exp(-1 * log_likelihood / num_target_tokens)
      result = {'language': language,
                'combined_idx': example['combined_idx'],
                'in_order_example_idx': example['in_order_example_idx'],
                'answer_idx': example['answer_idx'],
                'inputs': example['inputs'],
                'label': example['label'],
                'score': example['score'],
                'num_target_tokens': num_target_tokens,
                'loss': outputs.loss.item(),
                'll': log_likelihood,
                'perplexity_of_target': perplexity,
                }
      results.append(result)
      del input_ids
      del attention_mask
      del target_ids
    with open(f'{output_dir}/{dataset_name}-validation.json', 'w') as f:
      json.dump(results, f)
    torch.cuda.empty_cache()
