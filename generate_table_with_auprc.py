# =============================================================================
# Results 
# =============================================================================

results = {
    'int_val': {
        'amsterdam': {
            'mape':       80.3,   'mape_ci':  (79.342, 81.310),
            'auroc':       0.774,  'auroc_ci': (0.770,  0.778),
            'auprc':       0.300,  'auprc_ci': (0.293,  0.308), 'auprc_baseline': 0.113, 
        },
        'eicu': {
            'mape':       87.4,   'mape_ci':  (86.587, 88.220),
            'auroc':       0.788,  'auroc_ci': (0.783,  0.793),
            'auprc':       0.328,  'auprc_ci': (0.317,  0.338), 'auprc_baseline': 0.078, 
        },
        'hirid': {
            'mape':       83.2,   'mape_ci':  (82.012, 84.400),
            'auroc':       0.851,  'auroc_ci': (0.847,  0.856),
            'auprc':       0.444,  'auprc_ci': (0.430,  0.457), 'auprc_baseline': 0.101, 
        },
        'mimic4': {
            'mape':       82.8,   'mape_ci':  (81.977, 83.653),
            'auroc':       0.858,  'auroc_ci': (0.855,  0.860),
            'auprc':       0.481,  'auprc_ci': (0.474,  0.487), 'auprc_baseline': 0.161, 
        },
    },
    'ext_val': {
        'amsterdam': {
            'mape':       91.4,   'mape_ci':  (90.600, 92.223),
            'auroc':       0.738,  'auroc_ci': (0.736,  0.740),
            'auprc':       0.337,  'auprc_ci': (0.333,  0.342), 'auprc_baseline': 0.119,  
        },
        'eicu': {
            'mape':       98.0,   'mape_ci':  (97.445, 98.566),
            'auroc':       0.737,  'auroc_ci': (0.734,  0.740),
            'auprc':       0.330,  'auprc_ci': (0.326,  0.335), 'auprc_baseline': 0.139, 
        },
        'hirid': {
            'mape':       142.5,  'mape_ci':  (142.124, 142.884),
            'auroc':       0.801,  'auroc_ci': (0.799,   0.803),
            'auprc':       0.327,  'auprc_ci': (0.323,   0.331), 'auprc_baseline': 0.135,  
        },
        'mimic4': {
            'mape':       83.2,   'mape_ci':  (82.383, 84.055),
            'auroc':       0.771,  'auroc_ci': (0.768,  0.774),
            'auprc':       0.280,  'auprc_ci': (0.275,  0.285), 'auprc_baseline': 0.104, 
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
                 r'when training on subsets of the Blended-ICU dataset.}')
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

        int_str = fmt_auprc(iv['auprc'], iv['auprc_ci'], iv['auprc_baseline'])  # ← baseline ajoutée
        ext_str = fmt_auprc(ev['auprc'], ev['auprc_ci'], ev['auprc_baseline'])  # ← baseline ajoutée

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