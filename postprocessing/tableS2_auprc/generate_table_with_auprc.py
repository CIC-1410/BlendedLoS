# =============================================================================
# Results 
# =============================================================================

results = {
    'int_val': {
        'amsterdam': {
            'mape':       80.3,   'mape_ci':  (78.998, 81.646),
            'auroc':       0.774,  'auroc_ci': (0.770,  0.778),
            'auprc':       0.181,  'auprc_ci': (0.176,  0.186), 'auprc_baseline': 0.086,
        },
        'eicu': {
            'mape':       87.4,   'mape_ci':  (86.494, 88.334),
            'auroc':       0.788,  'auroc_ci': (0.783,  0.793),
            'auprc':       0.247,  'auprc_ci': (0.239,  0.256), 'auprc_baseline': 0.082,
        },
        'hirid': {
            'mape':       83.2,   'mape_ci':  (81.703, 84.747),
            'auroc':       0.851,  'auroc_ci': (0.847,  0.855),
            'auprc':       0.628,  'auprc_ci': (0.618,  0.638), 'auprc_baseline': 0.134,
        },
        'mimic4': {
            'mape':       82.8,   'mape_ci':  (82.005, 83.593),
            'auroc':       0.858,  'auroc_ci': (0.855,  0.861),
            'auprc':       0.543,  'auprc_ci': (0.537,  0.550), 'auprc_baseline': 0.169,
        },
    },
    'ext_val': {
        'amsterdam': {
            'mape':       91.4,   'mape_ci':  (90.418, 92.386),
            'auroc':       0.738,  'auroc_ci': (0.736,  0.740),
            'auprc':       0.404,  'auprc_ci': (0.399,  0.408), 'auprc_baseline': 0.135,
        },
        'eicu': {
            'mape':       98.0,   'mape_ci':  (97.379, 98.609),
            'auroc':       0.737,  'auroc_ci': (0.734,  0.740),
            'auprc':       0.292,  'auprc_ci': (0.288,  0.296), 'auprc_baseline': 0.140,
        },
        'hirid': {
            'mape':       142.5,  'mape_ci':  (141.903, 143.096),
            'auroc':       0.801,  'auroc_ci': (0.798,   0.804),
            'auprc':       0.323,  'auprc_ci': (0.318,   0.327), 'auprc_baseline': 0.136,
        },
        'mimic4': {
            'mape':       83.2,   'mape_ci':  (82.577, 83.835),
            'auroc':       0.771,  'auroc_ci': (0.768,  0.774),
            'auprc':       0.304,  'auprc_ci': (0.300,  0.309), 'auprc_baseline': 0.114,
        },
    },
}

SOURCES = {
    'amsterdam': 'AmsterdamUMC',
    'eicu':      'eICU',
    'hirid':     'HiRID',
    'mimic4':    'MIMIC-IV',
}

# =============================================================================
# Formatting helpers
# =============================================================================

def fmt_mape(val, ci):
    """Format MAPE: 80.3 [79.342 - 81.310]"""
    if val is None:
        return '--'
    return f'{val:.1f} [{ci[0]:.3f} -- {ci[1]:.3f}]'

def fmt_auroc(val, ci):
    """Format AUROC: 0.774 [0.770 - 0.778]"""
    if val is None:
        return '--'
    return f'{val:.3f} [{ci[0]:.3f} -- {ci[1]:.3f}]'

def fmt_auprc(val, ci, baseline):
    """Format AUPRC: 0.300 [0.293 - 0.308] (baseline: 0.123)"""
    if val is None:
        return '--'
    return f'{val:.3f} [{ci[0]:.3f} -- {ci[1]:.3f}] (baseline: {baseline:.3f})'

# =============================================================================
# LaTeX generation
# =============================================================================

def generate_latex():
    lines = []
    lines.append(r'\begin{table}[ht]')
    lines.append(r'\centering')
    lines.append(r'\caption{Comparison of the TPC model internal and external validation '
                 r'according to the dataset-models without drug exposure.}')
    lines.append(r'\label{tab:table2}')
    lines.append(r'\resizebox{\textwidth}{!}{%')
    lines.append(r'\begin{tabular}{llccccc}')
    lines.append(r'\toprule')
    lines.append(r'\textbf{Task} & \textbf{Training dataset} '
                 r'& \textbf{Internal validation} & \textbf{External validation} '
                 r'& \textbf{Variation} \\')
    lines.append(r'\midrule')

    # --- MAPE block ---
    lines.append(r'\multirow{4}{*}{\makecell[l]{Remaining Length\\of Stay prediction\\(MAPE) [95\% CI]}}')
    for i, (src_key, src_label) in enumerate(SOURCES.items()):
        iv = results['int_val'][src_key]
        ev = results['ext_val'][src_key]

        int_str  = fmt_mape(iv['mape'], iv['mape_ci'])
        ext_str  = fmt_mape(ev['mape'], ev['mape_ci'])

        # Variation between internal and external
        if iv['mape'] and ev['mape']:
            variation = (ev['mape'] - iv['mape']) / iv['mape'] * 100
            var_str = f'{variation:+.1f}\\%'
        else:
            var_str = '--'

        prefix = '' if i == 0 else ''
        lines.append(f'& {src_label} & {int_str} & {ext_str} & {var_str} \\\\')

    lines.append(r'\midrule')

    # --- AUROC block ---
    lines.append(r'\multirow{4}{*}{\makecell[l]{Mortality prediction\\(AUROC) [95\% CI]}}')
    for i, (src_key, src_label) in enumerate(SOURCES.items()):
        iv = results['int_val'][src_key]
        ev = results['ext_val'][src_key]

        int_str = fmt_auroc(iv['auroc'], iv['auroc_ci'])
        ext_str = fmt_auroc(ev['auroc'], ev['auroc_ci'])

        if iv['auroc'] and ev['auroc']:
            variation = (ev['auroc'] - iv['auroc']) / iv['auroc'] * 100
            var_str = f'{variation:+.1f}\\%'
        else:
            var_str = '--'

        lines.append(f'& {src_label} & {int_str} & {ext_str} & {var_str} \\\\')

    lines.append(r'\midrule')

    # --- AUPRC block ---
    lines.append(r'\multirow{4}{*}{\makecell[l]{Mortality prediction\\(AUPRC) [95\% CI]}}')
    for i, (src_key, src_label) in enumerate(SOURCES.items()):
        iv = results['int_val'][src_key]
        ev = results['ext_val'][src_key]

        int_str = fmt_auprc(iv['auprc'], iv['auprc_ci'], iv['auprc_baseline']) 
        ext_str = fmt_auprc(ev['auprc'], ev['auprc_ci'], ev['auprc_baseline'])  

        if iv['auprc'] and ev['auprc']:
            variation = (ev['auprc'] - iv['auprc']) / iv['auprc'] * 100
            var_str = f'{variation:+.1f}\\%'
        else:
            var_str = '--'

        lines.append(f'& {src_label} & {int_str} & {ext_str} & {var_str} \\\\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'}')  # end resizebox
    lines.append(r'\end{table}')

    return '\n'.join(lines)

# =============================================================================
# Output
# =============================================================================
latex = generate_latex()
print(latex)

# Save to file
with open('table_with_auprc.tex', 'w') as f:
    f.write(latex)
print('\nSauvegardé dans table2_with_auprc.tex')