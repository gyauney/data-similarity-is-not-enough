import numpy as np
import argparse
from collections import defaultdict
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import torch
import json
import os
from bigbench.api import task_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, type=str)
args = parser.parse_args()

if args.model == 't5-v1_1-xl':
  model_id = 'google/t5-v1_1-xl'
  model_filename_string = 't5-v1_1-xl'
  model_string = 'T5 v1.1 XL'
elif args.model == 't5':
  model_id = 't5-3b'
  model_filename_string = 't5-3b'
  model_string = 'T5 XL'

device = "cuda"
model = T5ForConditionalGeneration.from_pretrained(model_id).to(device)
model.eval()
tokenizer = T5Tokenizer.from_pretrained(model_id)

output_dir = f'./bigbench-output-{model_filename_string}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# load and tokenize the dataset
def get_multiple_choice_dataset(bb_dataset):
  dataset = load_dataset("bigbench", bb_dataset, split='default')
  # create new examples, each with a different target choice
  def concatenate_choices(example):
    batch = defaultdict(list)
    for i, (choice, score) in enumerate(zip(example['multiple_choice_targets'][0], example['multiple_choice_scores'][0])):
      batch['idx'].append(f'{example["idx"][0]}_{i}')
      batch['example_idx'].append(example["idx"][0])
      batch['answer_idx'].append(i)
      batch['inputs'].append(f'{example["inputs"][0]} {choice}')
      batch['score'].append(score)
      batch['choice'].append(choice)
      batch['targets'].append('')
      batch['multiple_choice_targets'].append('')
      batch['multiple_choice_scores'].append('')
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
  def tokenize_targets(example):
    choice_tokens = tokenizer(example['choice'], return_tensors="np", max_length=512, padding=False, truncation=True)
    example['choice_tokens'] = choice_tokens.input_ids
    return example
  dataset = dataset.map(concatenate_choices, batched=True, batch_size=1)
  dataset = dataset.remove_columns(["targets", "multiple_choice_targets", "multiple_choice_scores"])
  dataset = dataset.map(lambda examples: tokenizer(examples["inputs"], return_tensors="np", max_length=512, padding=True, truncation=True), batched=True, batch_size=1)
  dataset = dataset.map(mask_question_tokens, batched=False)
  dataset = dataset.map(tokenize_targets, batched=False)
  return dataset

def get_gen_dataset(bb_dataset):
  dataset = load_dataset("bigbench", bb_dataset, split='default')
  def mask_question_tokens_gen(example):
    # targets is a list
    example['combined_inputs_targets_list'] = []
    example['combined_inputs_target_input_ids_list'] = []
    example['combined_inputs_target_attention_mask_list'] = []
    example['target_ids_list'] = []
    example['num_target_tokens_list'] = []
    for target in example['targets']:
      combined_inputs_targets = f'{example["inputs"]} {target}'
      example['combined_inputs_targets_list'].append(combined_inputs_targets)
      combined_tokenized = tokenizer(combined_inputs_targets, return_tensors="np", max_length=512, padding=False, truncation=True)
      example['combined_inputs_target_input_ids_list'].append(combined_tokenized.input_ids)
      example['combined_inputs_target_attention_mask_list'].append(combined_tokenized.attention_mask)
      target_ids = np.array(combined_tokenized.input_ids.copy())
      target_tokens = tokenizer(target, return_tensors="np", max_length=512, padding=False, truncation=True)
      num_target_tokens = target_tokens.input_ids.shape[1]
      zero_idxs = np.where(np.array(combined_tokenized.attention_mask) == 0)[0]
      if len(zero_idxs) == 0:
        last_idx = combined_tokenized.input_ids.shape[1]
      else:
        last_idx = zero_idxs[0]
      example['num_target_tokens_list'].append(num_target_tokens)
      target_ids[:, :last_idx-num_target_tokens] = -100
      target_ids[:, last_idx:] = -100
      example['target_ids_list'].append(target_ids)
    return example
  # targets is a list of valid answers!! not just one
  dataset = dataset.remove_columns(["multiple_choice_targets", "multiple_choice_scores"])
  dataset = dataset.map(lambda examples: tokenizer(examples["inputs"], return_tensors="np", max_length=512, padding=True, truncation=True), batched=True, batch_size=1)
  dataset = dataset.map(mask_question_tokens_gen, batched=False)
  return dataset

bb_mul_datasets =  ['hindu_knowledge', 'bbq_lite_json', 'code_line_description', 'conceptual_combinations', 'emoji_movie', 'formal_fallacies_syllogisms_negation', 'known_unknowns', 'language_identification',
                    'logic_grid_puzzle', 'logical_deduction', 'misconceptions_russian', 'novel_concepts', 'play_dialog_same_or_different',
                    'strange_stories', 'strategyqa', 'symbol_interpretation', 'vitaminc_fact_verification', 'winowhy']
bb_gen_datasets = ['auto_debugging',
                   'conlang_translation', 'linguistics_puzzles', 'operators', 'parsinlu_reading_comprehension', 'repeat_copy_logic']

# all multiple choice tasks
for bb_dataset in bb_mul_datasets:
  print(bb_dataset)
  dataset = get_multiple_choice_dataset(bb_dataset)
  results = []
  for i in tqdm(range(len(dataset))):
    example = dataset[i]
    input_ids = torch.tensor([example['input_ids']]).to(device)
    attention_mask = torch.tensor([example['attention_mask']]).to(device)
    target_ids = torch.tensor(example['choice_tokens']).to(device)
    with torch.no_grad():
      outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)
    loss = outputs.loss.item()
    num_choice_tokens = example['num_choice_tokens']
    log_likelihood = -1 * loss * num_choice_tokens
    perplexity = np.exp(-1 * log_likelihood / num_choice_tokens)
    result = {'dataset': bb_dataset,
              'combined_idx': example['idx'],
              'example_idx': example['example_idx'],
              'answer_idx': example['answer_idx'],
              'score': example['score'],
              'num_choice_tokens': num_choice_tokens,
              'loss': outputs.loss.item(),
              'll': log_likelihood,
              'perp': perplexity,
              }
    results.append(result)
    del input_ids
    del attention_mask
    del target_ids
  with open(f'{output_dir}/{bb_dataset}.json', 'w') as f:
    json.dump(results, f)
  torch.cuda.empty_cache()
  
# all generative tasks
for bb_dataset in bb_gen_datasets:
  print(bb_dataset)
  dataset = get_gen_dataset(bb_dataset)
  results = []
  for i in tqdm(range(len(dataset))):
    example = dataset[i]
    input_ids = torch.tensor([example['input_ids']]).to(device)
    attention_mask = torch.tensor([example['attention_mask']]).to(device)
    # get the generation
    output_tokens = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=20)
    output = tokenizer.decode(output_tokens[0])
    # get the perplexity of the correct targets
    losses = []
    log_likelihoods = []
    num_target_tokens_list = []
    perplexities = []
    exact_str_matches = []
    for i, target in enumerate(example['targets']):
      metric = task_metrics.exact_str_match_fn([target], [output])
      combined_input_ids = torch.tensor(example['combined_inputs_target_input_ids_list'][i]).to(device)
      combined_attention_mask = torch.tensor(example['combined_inputs_target_attention_mask_list'][i]).to(device)
      target_ids = torch.tensor(example['target_ids_list'][i]).to(device)
      with torch.no_grad():
        outputs = model(combined_input_ids, attention_mask=combined_attention_mask, labels=target_ids)
      loss = outputs.loss.item()
      num_target_tokens = example['num_target_tokens_list'][i]
      log_likelihood = -1 * loss * num_target_tokens
      perplexity = np.exp(-1 * log_likelihood / num_target_tokens)
      exact_str_matches.append(metric['exact_str_match'])
      losses.append(loss)
      num_target_tokens_list.append(num_target_tokens)
      log_likelihoods.append(log_likelihood)
      perplexities.append(perplexity)
    # and finally get the loss/perplexity of just the input with no choice
    input_ids = torch.tensor([example['input_ids']]).to(device)
    attention_mask = torch.tensor([example['attention_mask']]).to(device)
    num_input_tokens = input_ids.shape[1]
    with torch.no_grad():
      outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
    log_likelihood_input = -1 * outputs.loss.item() * num_input_tokens
    perplexity_input = np.exp(-1 * log_likelihood_input / num_input_tokens)
    result = {'dataset': bb_dataset,
              'example_idx': example['idx'],
              'generated_output': output,
              'exact_str_match': exact_str_matches,
              'num_target_tokens_list': num_target_tokens_list,
              'loss_of_correct_targets': losses,
              'log_likelihood_of_correct_targets': log_likelihoods,
              'perplexity_of_correct_targets': perplexities,
              'log_likelihood_input': log_likelihood_input,
              'perplexity_of_input': perplexity_input,
              'num_input_tokens': num_input_tokens,
              }
    results.append(result)
    del input_ids
    del attention_mask
    del target_ids
    del output
  with open(f'{output_dir}/{bb_dataset}.json', 'w') as f:
    json.dump(results, f)
  torch.cuda.empty_cache()