import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy import stats
import math

# alpha: global false positive rate
def benjamini_yekutieli(p_values, alpha=0.05):
    num_tests = len(p_values)
    dependent_alpha = alpha / sum([1/i for i in range(1,num_tests)])
    fdr_vals = [k / num_tests * dependent_alpha for k in range(1, num_tests)]
    print('Significant?\tthreshold\tp-value')
    for p, v in zip(sorted(p_values), fdr_vals):
        print(f'{p < v}\t\t{v:.4f}\t\t{p:.4f}')

# The order matters to match the rows and columns of Tables 1 and 3 in the paper
rowcols = [('Average KL-divergence from the Pile (Hashed WPT bigrams)', 'Pythia 6.9B (0 shot) normalized aggregate score'),
           ('Average KL-divergence from C4 (Hashed WPT bigrams)', 'T5 v1.1 XL (0 shot) normalized aggregate score'),
           ('Average KL-divergence from C4 (Hashed WPT bigrams)', 'T5 v1.1 XL (2 shot) normalized aggregate score'),
           ('Average KL-divergence from C4 (Hashed WPT bigrams)', 'FLAN-T5 XL (0 shot) normalized aggregate score'),
           ('Average KL-divergence from C4 (Hashed WPT bigrams)', 'FLAN-T5 XL (2 shot) normalized aggregate score'),
           ('Average MAUVE score with the Pile', 'Pythia 6.9B (0 shot) normalized aggregate score'),
           ('Average MAUVE score with C4', 'T5 v1.1 XL (0 shot) normalized aggregate score'),
           ('Average MAUVE score with C4', 'T5 v1.1 XL (2 shot) normalized aggregate score'),
           ('Average MAUVE score with C4', 'FLAN-T5 XL (0 shot) normalized aggregate score'),
           ('Average MAUVE score with C4', 'FLAN-T5 XL (2 shot) normalized aggregate score'),
           ('Mean max cosine similarity with the Pile (mean across examples)', 'Pythia 6.9B (0 shot) normalized aggregate score'),
           ('Mean max cosine similarity with C4 (mean across examples)', 'T5 v1.1 XL (0 shot) normalized aggregate score'),
           ('Mean max cosine similarity with C4 (mean across examples)', 'T5 v1.1 XL (2 shot) normalized aggregate score'),
           ('Mean max cosine similarity with C4 (mean across examples)', 'FLAN-T5 XL (0 shot) normalized aggregate score'),
           ('Mean max cosine similarity with C4 (mean across examples)', 'FLAN-T5 XL (2 shot) normalized aggregate score'),
           ('Mean top 1000 cosine similarity with the Pile (mean across examples)', 'Pythia 6.9B (0 shot) normalized aggregate score'),
           ('Mean top 1000 cosine similarity with C4 (mean across examples)', 'T5 v1.1 XL (0 shot) normalized aggregate score'),
           ('Mean top 1000 cosine similarity with C4 (mean across examples)', 'T5 v1.1 XL (2 shot) normalized aggregate score'),
           ('Mean top 1000 cosine similarity with C4 (mean across examples)', 'FLAN-T5 XL (0 shot) normalized aggregate score'),
           ('Mean top 1000 cosine similarity with C4 (mean across examples)', 'FLAN-T5 XL (2 shot) normalized aggregate score'),
           ('Pythia 6.9B (0 shot) input with choices perplexity per token (mean across examples)', 'Pythia 6.9B (0 shot) normalized aggregate score'),
           ('T5 v1.1 XL (0 shot) input with choices perplexity per token (mean across examples)', 'T5 v1.1 XL (0 shot) normalized aggregate score'),
           ('T5 v1.1 XL (0 shot) input with choices perplexity per token (mean across examples)', 'T5 v1.1 XL (2 shot) normalized aggregate score'),
           ('FLAN-T5 XL (0 shot) input with choices perplexity per token (mean across examples)', 'FLAN-T5 XL (0 shot) normalized aggregate score'),
           ('FLAN-T5 XL (0 shot) input with choices perplexity per token (mean across examples)', 'FLAN-T5 XL (2 shot) normalized aggregate score'),
           ('Pythia 6.9B (0 shot) correct choice perplexity per token (mean across examples)', 'Pythia 6.9B (0 shot) normalized aggregate score'),
           ('T5 v1.1 XL (0 shot) correct choice perplexity per token (mean across examples)', 'T5 v1.1 XL (0 shot) normalized aggregate score'),
           ('T5 v1.1 XL (2 shot) correct choice perplexity per token (mean across examples)', 'T5 v1.1 XL (2 shot) normalized aggregate score'),
           ('FLAN-T5 XL (0 shot) correct choice perplexity per token (mean across examples)', 'FLAN-T5 XL (0 shot) normalized aggregate score'),
           ('FLAN-T5 XL (2 shot) correct choice perplexity per token (mean across examples)', 'FLAN-T5 XL (2 shot) normalized aggregate score'),
]

results = pd.read_csv('./raw-results/bigbench-raw-results-multiple-choice.csv')

print('Table 1:\n')
row_names = ['Bigram KL-divergence ($-$)', 'MAUVE score ($+$)', 'Max cosine similarity ($+$)', 'Mean cosine similarity ($+$)', 'Input perplexity ($-$)', 'Correct target perplexity ($-$)']
to_add = []
spearman_p_values = []
for i, (row, col) in enumerate(rowcols):
    x = results[row].tolist()
    y = results[col].tolist()
    res = spearmanr(results[row].tolist(), results[col].tolist())
    to_add.append(f'{res.statistic:.2f} & \\tiny{{{res.pvalue:.3f}}}')
    spearman_p_values.append(res.pvalue)
    # Print the current row
    if i % 5 == 4:
        print(row_names[math.floor(i / 5)] + ' & ' + ' & '.join(to_add) + '\\\\')
        to_add = []
print('\n--------------\n')


print('Table 3:\n')
row_names = ['Bigram KL-divergence ($-$)', 'MAUVE score ($+$)', 'Max cosine similarity ($+$)', 'Mean cosine similarity ($+$)', 'Input perplexity ($-$)', 'Correct target perplexity ($-$)']
to_add = []
for i, (row, col) in enumerate(rowcols):
    x = results[row].tolist()
    y = results[col].tolist()
    res = pearsonr(results[row].tolist(), results[col].tolist())
    to_add.append(f'{res.statistic:.2f} & \\tiny{{{res.pvalue:.3f}}}')
    # Print the current row
    if i % 5 == 4:
        print(row_names[math.floor(i / 5)] + ' & ' + ' & '.join(to_add) + '\\\\')
        to_add = []
print('\n--------------\n')

print('Benjamini-Yekutieli for Spearman correlations with scipy approximate p-values:\n')
benjamini_yekutieli(spearman_p_values)

print('\n--------------\n')

print('Alternate p-values from a permutation test (not in paper):\n')
spearman_p_values = []
for row, col in rowcols:
    x = results[row].tolist()
    y = results[col].tolist()
    def statistic(x):
        rs = stats.spearmanr(x, y).statistic  # ignore pvalue
        return rs
    res = stats.permutation_test((x,), statistic, alternative='two-sided',
                                    permutation_type='pairings')
    print(f'{row}, {col}: {res.statistic:.4f} ({res.pvalue:.6f})')
    spearman_p_values.append(res.pvalue)


print('\n--------------\n')

print('Benjamini-Yekutieli for Spearman correlations with permutation tests:\n')
benjamini_yekutieli(spearman_p_values)

