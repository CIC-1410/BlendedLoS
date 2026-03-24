# =============================================================================
# Results — Table S1: LSTM / Transformer / TPC on BlendedICU 
# =============================================================================

results = {
    'lstm': {
        'amsterdam': {
            'mape':      101.211, 'mape_ci':   (99.572,  102.319),
            'auroc':       0.764,  'auroc_ci':  (0.757,   0.771),
            'auprc':       0.296,  'auprc_ci':  (0.285,   0.307),  'auprc_baseline': 0.110,
        },
        'eicu': {
            'mape':       93.204,  'mape_ci':  (91.727,  94.6769),
            'auroc':       0.791,  'auroc_ci': (0.785,   0.797),
            'auprc':       0.342,  'auprc_ci': (0.329,   0.354),   'auprc_baseline': 0.117,
        },
        'hirid': {
            'mape':      100.064,  'mape_ci':  (98.043,  102.062),
            'auroc':       0.804,  'auroc_ci': (0.792,   0.816),
            'auprc':       0.456,  'auprc_ci': (0.434,   0.477),   'auprc_baseline': 0.073,
        },
        'mimic4': {
            'mape':       90.211,  'mape_ci':  (88.898,  91.553),
            'auroc':       0.755,  'auroc_ci': (0.749,   0.760),
            'auprc':       0.402,  'auprc_ci': (0.393,   0.411),   'auprc_baseline': 0.173,
        },
    },
    'transformer': {
        'amsterdam': {
            'mape':       95.868,  'mape_ci':  (94.435,  97.392),
            'auroc':       0.745,  'auroc_ci': (0.738,   0.752),
            'auprc':       0.318,  'auprc_ci': (0.306,   0.331),   'auprc_baseline': 0.122,
        },
        'eicu': {
            'mape':      100.911,  'mape_ci':  (99.200,  102.684),
            'auroc':       0.783,  'auroc_ci': (0.775,   0.791),
            'auprc':       0.335,  'auprc_ci': (0.321,   0.350),   'auprc_baseline': 0.095,
        },
        'hirid': {
            'mape':       93.285,  'mape_ci':  (91.359,  95.283),
            'auroc':       0.835,  'auroc_ci': (0.825,   0.846),
            'auprc':       0.309,  'auprc_ci': (0.288,   0.334),   'auprc_baseline': 0.066,
        },
        'mimic4': {
            'mape':       91.184,  'mape_ci':  (89.813,  92.581),
            'auroc':       0.795,  'auroc_ci': (0.791,   0.799),
            'auprc':       0.369,  'auprc_ci': (0.360,   0.378),   'auprc_baseline': 0.171,
        },
    },
    'tpc': {
        'amsterdam': {
            'mape':       84.081,  'mape_ci':  (82.672,  85.491),
            'auroc':       0.846,  'auroc_ci': (0.841,   0.851),
            'auprc':       0.350,  'auprc_ci': (0.337,   0.364),   'auprc_baseline': 0.098,
        },
        'eicu': {
            'mape':       88.073,  'mape_ci':  (86.715,  89.448),
            'auroc':       0.597,  'auroc_ci': (0.589,   0.605),
            'auprc':       0.219,  'auprc_ci': (0.211,   0.229),   'auprc_baseline': 0.130,
        },
        'hirid': {
            'mape':       86.518,  'mape_ci':  (84.448,  88.639),
            'auroc':       0.887,  'auroc_ci': (0.881,   0.894),
            'auprc':       0.649,  'auprc_ci': (0.632,   0.666),   'auprc_baseline': 0.140,
        },
        'mimic4': {
            'mape':       75.646,  'mape_ci':  (74.701,  76.611),
            'auroc':       0.788,  'auroc_ci': (0.783,   0.793),
            'auprc':       0.379,  'auprc_ci': (0.369,   0.389),   'auprc_baseline': 0.145,
        },
    },
}

SOURCES = {
    'amsterdam': 'AmsterdamUMC',
    'eicu':      'eICU',
    'hirid':     'HiRID',
    'mimic4':    'MIMIC-IV',
}

MODELS = ['lstm', 'transformer', 'tpc']
MODEL_LABELS = {'lstm': 'LSTM', 'transformer': 'Transformer', 'tpc': 'TPC'}

# =============================================================================
# Formatting helpers
# =============================================================================

def fmt_mape(val, ci):
    """Format MAPE: 101.211 [99.572 - 102.319]"""
    if val is None:
        return '--'
    return f'{val:.3f} [{ci[0]:.3f} -- {ci[1]:.4f}]'

def fmt_auroc(val, ci):
    """Format AUROC: 0.764 [0.757 - 0.771]"""
    if val is None:
        return '--'
    return f'{val:.3f} [{ci[0]:.3f} -- {ci[1]:.3f}]'

def fmt_auprc(val, ci, baseline):
    """Format AUPRC: 0.300 [0.293 - 0.308] (baseline: 0.113)"""
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
    lines.append(r'\caption{Performance comparison of the LSTM, Transformer, and TPC '
                 r'architectures when training on 12,654 patients from the Blended-ICU dataset.}')
    lines.append(r'\label{tab:table_s1}')
    lines.append(r'\resizebox{\textwidth}{!}{%')
    lines.append(r'\begin{tabular}{llccc}')
    lines.append(r'\toprule')
    lines.append(
        r'\textbf{Prediction} & \textbf{Validation dataset}'
        r' & \textbf{LSTM} & \textbf{Transformer} & \textbf{TPC} \\'
    )
    lines.append(r'\midrule')

    # --- MAPE block ---
    lines.append(
        r'\multirow{4}{*}{\makecell[l]{Remaining Length\\of Stay,\\MAPE {[}95\% CI{]}}}'
    )
    for src_key, src_label in SOURCES.items():
        cols = [
            fmt_mape(results[m][src_key]['mape'], results[m][src_key]['mape_ci'])
            for m in MODELS
        ]
        lines.append(f'& {src_label} & {cols[0]} & {cols[1]} & {cols[2]} \\\\')

    lines.append(r'\midrule')

    # --- AUROC block ---
    lines.append(
        r'\multirow{4}{*}{\makecell[l]{Mortality,\\AUROC {[}95\% CI{]}}}'
    )
    for src_key, src_label in SOURCES.items():
        cols = [
            fmt_auroc(results[m][src_key]['auroc'], results[m][src_key]['auroc_ci'])
            for m in MODELS
        ]
        lines.append(f'& {src_label} & {cols[0]} & {cols[1]} & {cols[2]} \\\\')

    lines.append(r'\midrule')

    # --- AUPRC block ---
    lines.append(
        r'\multirow{4}{*}{\makecell[l]{Mortality,\\AUPRC {[}95\% CI{]}}}'
    )
    for src_key, src_label in SOURCES.items():
        cols = [
            fmt_auprc(
                results[m][src_key]['auprc'],
                results[m][src_key]['auprc_ci'],
                results[m][src_key]['auprc_baseline'],
            )
            for m in MODELS
        ]
        lines.append(f'& {src_label} & {cols[0]} & {cols[1]} & {cols[2]} \\\\')

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

with open('table_s1.tex', 'w') as f:
    f.write(latex)
print('\nSauvegardé dans table_s1.tex')