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
from bigbench.api import task_metrics
import time

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

output_dir = f'./bigbench-output-{model_filename_string}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print('loaded the model!')

dataset_to_avg_random_guess = {}

data_dir = './bigbench-datasets'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
if not os.path.exists('./subsampled-idxs'):
    os.makedirs('./subsampled-idxs')
    

# load and tokenize the dataset
def get_multiple_choice_dataset(bb_dataset, subsample, num_to_subsample=None, num_shots=None):
  dataset = load_dataset("bigbench", bb_dataset, split='default')
  few_shot_prefix = ''

  # Get the accuracy from random guessing
  # To calculate normalized aggregate scores
  random_guesses = [1 / len(d) for d in dataset['multiple_choice_targets']]
  avg_random_guess = sum(random_guesses)/len(random_guesses)
  dataset_to_avg_random_guess[bb_dataset] = avg_random_guess

  # few-shot
  if num_shots:
    idxs_few_shot = random.sample(range(len(dataset)), num_shots)
    examples_few_shot = [dataset[i] for i in idxs_few_shot]
    few_shot_prefix = get_few_shot_prefix(examples_few_shot)
    print(f'Few shot prefix: {few_shot_prefix}')
    idxs_to_sample_from = set(range(len(dataset))) - set(idxs_few_shot)
  else:
    idxs_to_sample_from = set(range(len(dataset)))

  # First subsample the dataset
  if subsample and num_to_subsample < (len(dataset) - num_shots):
    print(f'{bb_dataset}: sampling {num_to_subsample} out of {len(dataset)} examples.')
    subsampled_idxs_filename = f'./subsampled-idxs/{bb_dataset}_{num_to_subsample}-examples_idxs.json'
    if os.path.exists(subsampled_idxs_filename):
      with open(subsampled_idxs_filename, 'r') as f:
        idxs_to_subsample = json.load(f)
    else:
      idxs_to_subsample = random.sample(idxs_to_sample_from, num_to_subsample)
      with open(subsampled_idxs_filename, 'w') as f:
        json.dump(idxs_to_subsample, f)
    dataset = dataset.select(idxs_to_subsample)
  else:
    # we're not subsampling the dataset but we might need to 
    # remove examples used for few-shot demonstrations if they exist
    dataset = dataset.select(idxs_to_sample_from)


  def get_few_shot_prefix(few_shot_examples):
    prefix = ''
    for example in few_shot_examples:
      for i, (choice, score) in enumerate(zip(example['multiple_choice_targets'], example['multiple_choice_scores'])):
        if score:
          prefix += f'{example["inputs"]} {choice}\n'
    return prefix
  # create new examples, each with a different target choice
  def concatenate_choices(example):
    batch = defaultdict(list)
    for i, (choice, score) in enumerate(zip(example['multiple_choice_targets'][0], example['multiple_choice_scores'][0])):
      batch['idx'].append(f'{example["idx"][0]}_{i}')
      batch['example_idx'].append(example["idx"][0])
      batch['answer_idx'].append(i)
      batch['inputs'].append(f'{few_shot_prefix}{example["inputs"][0]} {choice}')
      batch['inputs_without_choice'].append(example["inputs"][0])
      batch['domain_prefix_choice'].append(f'{example["domain_prefix"][0]}{choice}')
      batch['score'].append(score)
      batch['choice'].append(choice)
      batch['targets'].append('')
      batch['multiple_choice_targets'].append('')
      batch['multiple_choice_scores'].append('')
      batch['domain_prefix'].append('')
    return batch
  # misconceptions_russian has no input--the choices are the complete example
  def concatenate_choices_misconceptions_russian(example):
    batch = defaultdict(list)
    for i, (choice, score) in enumerate(zip(example['multiple_choice_targets'][0], example['multiple_choice_scores'][0])):
      batch['idx'].append(f'{example["idx"][0]}_{i}')
      batch['example_idx'].append(example["idx"][0])
      batch['answer_idx'].append(i)
      batch['inputs'].append(choice)
      batch['inputs_without_choice'].append(choice)
      batch['domain_prefix_choice'].append(f'{example["domain_prefix"][0]}{choice}')
      batch['score'].append(score)
      batch['choice'].append(choice)
      batch['targets'].append('')
      batch['multiple_choice_targets'].append('')
      batch['multiple_choice_scores'].append('')
      batch['domain_prefix'].append('')
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
  dataset = dataset.add_column("domain_prefix", [bb_dataset_to_domain_prefix[bb_dataset]] * len(dataset))
  if bb_dataset == 'misconceptions_russian':
    dataset = dataset.map(concatenate_choices_misconceptions_russian, batched=True, batch_size=1)
  else:
    dataset = dataset.map(concatenate_choices, batched=True, batch_size=1)
  dataset = dataset.remove_columns(["targets", "multiple_choice_targets", "multiple_choice_scores", "domain_prefix"])
  dataset = dataset.map(lambda examples: tokenizer(examples["inputs"], return_tensors="np", max_length=512, padding=True, truncation=True), batched=True, batch_size=1)
  dataset = dataset.map(tokenize_inputs_without_choice, batched=False)
  dataset = dataset.map(tokenize_domain_prefix, batched=False)
  dataset = dataset.map(mask_question_tokens, batched=False)
  dataset = dataset.map(mask_domain_prefix_tokens, batched=False)

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

def save_texts(bb_dataset, dataset):
  examples = [{'id': e['example_idx'], 'text': e['inputs']} for e in dataset if e['score']]
  with open(f'{data_dir}/{bb_dataset}.json', 'w') as f:
    json.dump(examples, f, ensure_ascii=False)

bb_mul_datasets =  ['hindu_knowledge', 'bbq_lite_json', 'code_line_description', 'conceptual_combinations', 'emoji_movie', 'formal_fallacies_syllogisms_negation', 'known_unknowns', 'language_identification',
                    'logic_grid_puzzle', 'logical_deduction', 'misconceptions_russian', 'novel_concepts', 'play_dialog_same_or_different',
                    'strange_stories', 'strategyqa', 'symbol_interpretation', 'vitaminc_fact_verification', 'winowhy']
bb_gen_datasets = ['auto_debugging',
                   'conlang_translation', 'linguistics_puzzles', 'operators', 'parsinlu_reading_comprehension', 'repeat_copy_logic']
bb_dataset_to_domain_prefix = {
  'hindu_knowledge': 'A: ',
  'bbq_lite_json': 'A: ',
  'code_line_description': 'English language description: ',
  'conceptual_combinations': 'Answer: ',
  'emoji_movie': 'A: ',
  'formal_fallacies_syllogisms_negation': 'A: ',
  'known_unknowns': 'A: ',
  'language_identification': 'Language: ',
  'logic_grid_puzzle': 'A: ',
  'logical_deduction': '. ',
  'misconceptions_russian': '', # each answer choice is a full statement
  'novel_concepts': 'A: ',
  'play_dialog_same_or_different': 'A: ',
  'strange_stories': 'A: ',
  'strategyqa': 'A: ',
  'symbol_interpretation': 'A: ',
  'vitaminc_fact_verification': 'True, False, or Neither? ',
  'winowhy': 'The above reasoning is ',
} 

times = {}

# all multiple choice tasks
for bb_dataset in bb_mul_datasets:
  print(bb_dataset)
  dataset = get_multiple_choice_dataset(bb_dataset, subsample=False)
  save_texts(bb_dataset, dataset)
  start = time.time()
  results = []
  for i in tqdm(range(len(dataset))):
    example = dataset[i]
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
    result = {'dataset': bb_dataset,
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
  times[bb_dataset] = end - start
  with open(f'{output_dir}/{bb_dataset}.json', 'w') as f:
    json.dump(results, f)
  torch.cuda.empty_cache()


# all generative tasks
for bb_dataset in bb_gen_datasets:
  print(bb_dataset)
  dataset = get_gen_dataset(bb_dataset)
  save_texts(bb_dataset, dataset)
  results = []
  for i in tqdm(range(len(dataset))):
    example = dataset[i]
    input_ids = torch.tensor([example['input_ids']]).to(device)
    attention_mask = torch.tensor([example['attention_mask']]).to(device)
    # get the generation
    output_tokens = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=20)
    output = tokenizer.decode(output_tokens[0])
    if i < 5:
      print(example['inputs'])
      print(example['targets'])
      print(output)
      print()
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

print(json.dumps(dataset_to_avg_random_guess, indent=4))

with open(f'{output_dir}/times.json', 'w') as f:
    json.dump(times, f)