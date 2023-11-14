# data-similarity-is-not-enough

Code and data for "Data Similarity is Not Enough to Explain Language Model
Performance" (EMNLP 2023).

## Requirements

The file `requirements.txt` contains the Python packages required to run this code.
This code is written for use with a GPU.

## Raw results

This repository contains the raw similarity and performance results used to 
construct the figures and tables in the paper:

- `raw-results/bigbench-raw-results-multiple-choice.csv`:
  overall BIG-bench Lite performances and similarities used to create the figures and tables
- `raw-results/cosine-similarities`: This directory contains the cosine
  similarities between Sentence-T5 embeddings of sampled downstream documents
  and **all** of the Pile and C4.
- `raw-results/sampled-pretraining-idxs`: This directory contains the indices of sampled
  pretraining documents from the Pile and C4. These correspond to indices in the
  datasets when loaded using Hugging Face `datasets`. They were used for
  calculating KL-divergences and MAUVE scores against samples of the pretraining
  datasets.

The `raw-results` directory also contains the performances and similarities for
Stack Exchange and XNLI datasets.

## Reproducing figures and tables from the paper with provided results

To reproduce the figures and tables from the paper, run the following commands:

- `python tables-1-and-3.py`: generates the tables with correlations between
  similarity and performance for BIG-bench Lite multiple choice tasks
- `python table-2.py`: generates the table with correlations between similarity
  measures when comparing BIG-bench Lite with C4 and the Pile
- `python figure-1.py --reproduction`: generates the graphs of similarity and
  performance for Stack Exchange and BIG-bench Lite multiple choice tasks.
- `python figure-2.py --reproduction`: generates the graphs of similarity and
  performance for Stack Exchange and XNLI.
- `python figure-bigbench-violinplots.py --model Pythia-6.9B`:
  generates the BIG-bench Lite violinplots in figure 3
- `python figure-4.py`:
  generates the GLUE violinplots in figure 4
- `python figure-5.py --reproduction`: generates small multiples for all XNLI
  languages
- `python figure-6.py`: generates the BIG-bench Lite barplots binned by
  similarity quartile
- `python figure-bigbench-violinplots.py --model T5`:
  generates the BIG-bench Lite violinplots in figure 7
- `python figure-8.py`: generates the Stack Exchange AM/PM graph of similarity
  and performance

--------

## Datasets

### Stack Exchange classification

Each file contains 1000 posts from Stack Exchange: 500 posts from each of the `bicycles` and `cstheory` Stack Exchange forums. All posts were originally in English. The percentage in the filename refers to how much of each post has been translated by Google Translate. For example, the file `bicycles-cstheory_50-percent-finnish.json` contains posts whose first half is in Finnish. The original dataset is in `bicycles-cstheory_100-percent-english.json`.

These datasets were adapted from [Comparing Text Representations: A Theory-Driven Approach (EMNLP 2021)](https://aclanthology.org/2021.emnlp-main.449.pdf).

### XNLI

Each file contains 2,500 natural language inference questions. All documents were originally in English. Again, the percentage in the filename refers to how much of each document has been translated by Google Translate. For example, the file `xnli_50-percent-spanish` contains documents whose first half is in Spanish. The original XNLI datasets are those that have `100-percent` in the filename.

These datasets were adapted from [XNLI: Evaluating cross-lingual sentence representations (EMNLP 2018)](https://aclanthology.org/D18-1269.pdf).

### Prompt templates

**Stack Exchange forum classification**: Each example contains the post text
and the forum it was posted to (either 'bicycles' or 'cstheory'). We format
each example as: `{text}\nThis post is about {forum}`.
The prompt is the same across languages.

**Stack Exchange AM/PM classification**: Each example contains the post text
and whether it was posted in the 'morning' or 'afternoon'. We format
each example as: `{text}\nThis was posted in the {label}`.
The prompt is the same across languages.

**XNLI**: Since XNLI consists of translations of MNLI, we use an MNLI prompt
for all datasets. Examples include a premise, a hypothesis, and a label.
Labels are mapped, in order, from 'entailment', 'neutral',
'contradiction' to 'Yes', 'Maybe', 'No'. An example then becomes:
`{premise} {label}, {hypothesis}`.
The prompt is the same across languages.

**BIG-bench Lite**: BIG-bench Lite examples come pre-formatted when loaded
through the Hugging Face `datasets` library. We use the existing formatting.

**GLUE**: For T5 experiments, we use the prompting style described in the 
[original T5 paper](https://www.jmlr.org/papers/volume21/20-074/20-074.pdf)
(Raffel et al., JMLR 2020). For example, an MNLI example with a premise and a
hypothesis becomes: `'mnli hypothesis: {hypothesis} premise: {premise} {label}'`
where the label can be 'entailment', 'contradiction', or 'neutral'.

----

## Code

This code calculates zero-shot accuracy on BIG-bench Lite, GLUE, Stack Exchange,
and XNLI. It then calculates each downstream dataset's similarity to two
pretraining datasets (the Pile and C4). 

### 1. Download pretraining datasets. Sample them and create token distributions and ST5 embeddings.

**```sample_pretraining_data.py```:** 
Downloads C4 and the Pile, samples 100,000 docs at a time, and creates ST5 embeddings.
Arguments:

- `dataset_name`: can be either `the_pile` or `c4`
- `sample_num`: an integer identifier for this sample--we use integers from 1 to 8
- `num_samples`: the number of documents to sample

**`n-gram-distributions-pretraining.py`:**
Generate token distributions for samples from C4 and the Pile. Arguments:

- `dataset_name`: can be either `the_pile` or `c4`
- `sample_num`: an integer identifier for this sample--we use integers from 1 to 8

### 2. Evaluate downstream datasets

**`eval-stackexchange.py`:**
Run Pythia-6.9B zero-shot eval on Stack Exchange classification tasks.
Arguments:

- `label_type`: Can be either `forum` or `ampm`.
- `model`: Name of a `transformers` model. We report results with `Pythia-6.9B`.

**`eval-xnli.py`:**
Run Pythia-6.9B zero-shot eval on XNLI classification tasks.
Arguments:

- `model`: Name of a `transformers` model. We report results with `Pythia-6.9B`.

**`eval-bigbench.py`:**
Run Pythia-6.9B zero-shot and few-shot eval on BIG-bench Lite tasks.
Arguments:

- `model`: Name of a `transformers` model. We report results with `Pythia-6.9B`.

**`eval-bigbench-t5.py`:**
Run T5-3B and T5 v1.1 XL zero-shot and few-shot eval on BIG-bench Lite tasks.
Arguments:

- `model`: Name of a `transformers` model.
  Options are `t5` for finetuned T5-3B and `t5-v1_1-xl` for T5 v1.1 XL.

**`eval-glue.py`:**
Run zero-shot eval on GLUE tasks.

- `model`: Name of a `transformers` model.

### 3. Get token distributions and embeddings for downstream tasks.

Run the following files to create token distributions:

- `n-gram-distributions-bigbench.py`
- `n-gram-distributions-stackexchange.py`
- `n-gram-distributions-xnli.py`

Run the following file to construct embeddings:

- `embeddings-bigbench.py`
- `embeddings-glue.py`.

### 4. Process downstream results and calculate similarities.

Run the following files:

- `process-stackexchange.py`
- `process-xnli.py`
- `process-bigbench.py`
- `process-glue.py`

We are unable to include our code for calculating cosine similarities between
entire pretraining datasets and examples from downstream datasets, but the
scripts in this section calculate such similarities against a sample of the
pretraining dataset. We include our raw results against the entire pretraining
dataset in the `cosine-similarities` directory.

### 5. Make figures

Run the following files to generate the tables and figures in the paper:

- `tables-1-and-3.py`: Generates tables 1 and 3.
- `table-2.py`: Generates tables 2.
- `figure-1.py`: Generates figure 1.
- `figure-2.py`: Generates figure 2.
- `figure-bigbench-violinplots.py`: Generates figures 3 and 7.
  Arguments are `model` (either `Pythia-6.9B` or `T5`)
  and `similarity_type` (either `max` or `mean`).
- `figure-4.py`: Generates figure 4.
- `figure-5.py`: Generates figure 5.
- `figure-6.py`: Generates figure 6.
  Arguments are `model` (either `Pythia-6.9B` or `T5`).
- `process-stackexchange-ampm.py`: Generates figure 8.

------

## Raw performance results

This table contains the raw performance results for BIG-bench Lite multiple
choice tasks that are used to construct tables and figures but that are not
explicitly included in the main paper.

| Dataset | Pythia-6.9B (0 shot) | T5 v1.1 XL (0 shot) | T5 v1.1 XL (2 shot) | Flan-T5 XL (0 shot) | Flan-T5 XL (2 shot) |
|---------|----------------------|---------------------|---------------------|---------------------|---------------------|
| bbq_lite_json | -22.02 | 4.80 | 8.75 | 19.90 | 37.76 |
| code_line_description | -2.00 | -10.86 | -10.86 | 17.96 | 24.61 |
| conceptual_combinations | 0.00 | 0.32 | 5.50 | 52.10 | 41.75 |
| emoji_movie | 6.25 | -1.25 | -5.00 | 3.75 | 3.75 |
| formal_fallacies_syllogisms_negation | -0.07 | 0.00 | 0.00 | 1.18 | 2.89 |
| hindu_knowledge | 1.96 | -0.51 | 1.02 | 14.72 | 16.24 |
| known_unknowns | 4.35 | 0.00 | 4.35 | -8.70 | -17.39 |
| language_identification | 5.49 | 0.25 | -0.08 | 12.34 | 4.73 |
| logic_grid_puzzle | 18.88 | -2.60 | -2.60 | 8.43 | 0.01 |
| logical_deduction | 0.00 | -1.04 | 0.16 | 15.91 | 11.09 |
| novel_concepts | -8.53 | -21.09 | -17.19 | -9.38 | -1.56 |
| operators | 0.00 | 0.95 | 0.00 | 5.71 | 6.19 |
| play_dialog_same_or_different | 21.94 | -26.16 | -26.16 | 18.75 | 15.63 |
| strange_stories | -3.02 | -9.56 | -5.79 | 46.16 | 34.83 |
| strategyqa | 3.89 | -6.42 | -6.51 | 7.56 | 28.35 |
| symbol_interpretation | -1.01 | -0.51 | -2.40 | 1.01 | 1.01 |
| vitaminc_fact_verification | 24.57 | 5.34 | 5.34 | 56.71 | 41.31 |
