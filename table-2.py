import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy import stats

results = pd.read_csv('./raw-results/bigbench-raw-results-multiple-choice.csv')

name_to_display_name = {
    'Average MAUVE score with C4': 'MAUVE score',
    'Mean max cosine similarity with C4 (mean across examples)': 'Max cosine similarity',
    'Mean max cosine similarity with the Pile (mean across examples)': 'Max cosine similarity',
    'Average MAUVE score with the Pile': 'MAUVE score',
    'Average MAUVE score with C4': 'MAUVE score',
    'T5 v1.1 XL (0 shot) input with choices perplexity per token (mean across examples)': 'Input perplexity',
    'T5 v1.1 XL (2 shot) correct choice perplexity per token (mean across examples)': 'Correct target perplexity',
    'Pythia 6.9B (0 shot) input with choices perplexity per token (mean across examples)': 'Input perplexity',
    'Pythia 6.9B (0 shot) correct choice perplexity per token (mean across examples)': 'Correct target perplexity',
}

# print the C4 table
print('C4\n')

# This order is required to match up with the rows/columns in Table 2
rows = ['Average MAUVE score with C4',
        'Mean max cosine similarity with C4 (mean across examples)',
        'T5 v1.1 XL (0 shot) input with choices perplexity per token (mean across examples)',
        'T5 v1.1 XL (2 shot) correct choice perplexity per token (mean across examples)'
        ]
cols = ['Average KL-divergence from C4 (Hashed WPT bigrams)',
        'Average MAUVE score with C4',
        'Mean max cosine similarity with C4 (mean across examples)',
        'T5 v1.1 XL (0 shot) input with choices perplexity per token (mean across examples)',
        ]

# print the table in the right order
for i, row in enumerate(rows):
    line = ''
    to_add = []
    for j, col in enumerate(cols):
        if j <= i:
            res = spearmanr(results[row].tolist(), results[col].tolist())
            to_add.append(f'{res.statistic:.2f} & \\tiny{{{res.pvalue:.3f}}}')
        else:
            # Gray out this square--it's a repeat
            to_add.append('\multicolumn{2}{c}{\cellcolor{gray!15}}')
    line = name_to_display_name[row] + ' & ' + ' & '.join(to_add) + '\\\\'
    print(line)

print('\n--------------\n')

# Print the Pile table
print('The Pile\n')

# This order is required to match up with the rows/columns in Table 2
rows = ['Average MAUVE score with the Pile',
        'Mean max cosine similarity with the Pile (mean across examples)',
        'Pythia 6.9B (0 shot) input with choices perplexity per token (mean across examples)',
        'Pythia 6.9B (0 shot) correct choice perplexity per token (mean across examples)'
        ]
cols = ['Average KL-divergence from the Pile (Hashed WPT bigrams)',
        'Average MAUVE score with the Pile',
        'Mean max cosine similarity with the Pile (mean across examples)',
        'Pythia 6.9B (0 shot) input with choices perplexity per token (mean across examples)',
        ]

for i, row in enumerate(rows):
    line = ''
    to_add = []
    for j, col in enumerate(cols):
        if j <= i:
            res = spearmanr(results[row].tolist(), results[col].tolist())
            to_add.append(f'{res.statistic:.2f} & \\tiny{{{res.pvalue:.3f}}}')
        else:
            # Gray out this square--it's a repeat
            to_add.append('\multicolumn{2}{c}{\cellcolor{gray!15}}')
    line = name_to_display_name[row] + ' & ' + ' & '.join(to_add) + '\\\\'
    print(line)

print('\n--------------\n')
print('Spearman correlations between cosine similarities (for paper appendix)\n')


res = spearmanr(results['Mean max cosine similarity with C4 (mean across examples)'].tolist(),
                results['Mean top 1000 cosine similarity with C4 (mean across examples)'].tolist())
print(f'C4: {res.statistic:.2f} ({res.pvalue})')

res = spearmanr(results['Mean max cosine similarity with the Pile (mean across examples)'].tolist(),
                results['Mean top 1000 cosine similarity with the Pile (mean across examples)'].tolist())
print(f'The Pile: {res.statistic:.2f} ({res.pvalue})')
