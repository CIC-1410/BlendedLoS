# =============================================================================
# Results 
# =============================================================================

results = {
    'int_val': {
        'amsterdam': {
            'mape':       80.3,   'mape_ci':  (79.500, 81.1),
            'auroc':       0.774,  'auroc_ci': (0.770,  0.778),
            'auprc':       0.304,  'auprc_ci': (0.296,  0.312), 'auprc_baseline': 0.113,
        },
        'eicu': {
            'mape':       87.4,   'mape_ci':  (86.8, 88.0),
            'auroc':       0.788,  'auroc_ci': (0.783,  0.793),
            'auprc':       0.355,  'auprc_ci': (0.344,  0.365), 'auprc_baseline': 0.078,
        },
        'hirid': {
            'mape':       83.2,   'mape_ci':  (82.1, 84.1),
            'auroc':       0.851,  'auroc_ci': (0.847,  0.856),
            'auprc':       0.441,  'auprc_ci': (0.427,  0.455), 'auprc_baseline': 0.101,
        },
        'mimic4': {
            'mape':       87.228, 'mape_ci':  (86.319, 88.129),
            'auroc':       0.897,  'auroc_ci': (0.895,  0.900),
            'auprc':       0.456,  'auprc_ci': (0.448,  0.464), 'auprc_baseline': 0.108,
        },
    },
    'ext_val': {
        'amsterdam': {
            'mape':       121.858, 'mape_ci':  (120.910, 122.833),
            'auroc':       0.808,   'auroc_ci': (0.805,   0.810),
            'auprc':       0.294,   'auprc_ci': (0.289,   0.299), 'auprc_baseline': 0.091,
        },
        'eicu': {
            'mape':       100.144, 'mape_ci':  (99.547, 100.761),
            'auroc':       0.768,   'auroc_ci': (0.766,   0.771),
            'auprc':       0.269,   'auprc_ci': (0.265,   0.273), 'auprc_baseline': 0.107,
        },
        'hirid': {
            'mape':       82.975,  'mape_ci':  (82.570, 83.371),
            'auroc':       0.777,   'auroc_ci': (0.775,  0.779),
            'auprc':       0.331,   'auprc_ci': (0.327,  0.336), 'auprc_baseline': 0.115,
        },
        'mimic4': {
            'mape':       109.45,  'mape_ci':  (108.633, 110.266),
            'auroc':       0.773,   'auroc_ci': (0.770,   0.776),
            'auprc':       0.270,   'auprc_ci': (0.265,   0.274), 'auprc_baseline': 0.104,
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