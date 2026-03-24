import pandas as pd
import numpy as np
import re
from sklearn import metrics
from sklearn.utils import resample
from rocauc_comparison import delong_roc_variance


# =============================================================================
# Helper functions
# =============================================================================

def calculate_mape(y_true, y_pred):
    return ((y_true - y_pred).abs() / y_true).mean() * 100


def calculate_composite(mape, mape_ref, auc, auc_ref):
    return 0.5 * (mape / mape_ref + (1 - auc) / (1 - auc_ref))


def load_predictions(repo):
    """Load and merge LOS + mortality prediction CSVs for a given repo."""
    base_tl  = "Y:/DDS_Rocheteau/BlendedLOS/results_transferlearning/main_experiment/"
    base_srv = "Y:/DDS_Rocheteau/BlendedLOS/results_mimic4update-server/main_experiment/"

    if repo in ("mimic4_75", "amsterdam_25_retrain_all_layers"):
        base = base_tl + repo + "/multitask/TPC/26-03-23_183012/"
    else:
        base = base_srv + repo + "/multitask/TPC/26-03-20_212503/"

    df_los  = pd.read_csv(base + "test_predictions_los.csv")
    df_mort = pd.read_csv(base + "test_predictions_mort.csv")

    df_mort["pred_los"]  = df_los["pred_los"]
    df_mort["label_los"] = df_los["label"]
    df_mort.rename(columns={"label": "label_mort"}, inplace=True)

    return df_mort


def filter_by_source(df, source):
    """Keep only rows matching the given source(s)."""
    if isinstance(source, str):
        pattern = source
    else:
        pattern = "|".join(map(re.escape, source))
    return df[df["patientids"].str.contains(pattern, case=False, na=False)]


def bootstrap_metrics(df, df_alive, mape_ref, auc_ref, n_bootstrap=1000):
    """
    Bootstrap MAPE, AUC and composite metric.
    df       : tous les patients (pour AUC)
    df_alive : patients survivants uniquement (pour MAPE)
    Resample chaque DataFrame séparément car les populations sont différentes.
    """
    mape_boot      = np.zeros(n_bootstrap)
    auc_boot       = np.zeros(n_bootstrap)
    composite_boot = np.zeros(n_bootstrap)

    for j in range(n_bootstrap):
        # Rééchantillonner le DataFrame entier pour garder l'alignement patient
        df_boot = resample(df, random_state=j)

        fpr, tpr, _ = metrics.roc_curve(df_boot["label_mort"], df_boot["pred_mort"], pos_label=1)
        auc_boot[j]       = metrics.auc(fpr, tpr)
        df_alive_boot     = resample(df_alive, random_state=j)
        mape_boot[j]      = calculate_mape(df_alive_boot["label_los"], df_alive_boot["pred_los"])
        composite_boot[j] = calculate_composite(mape_boot[j], mape_ref, auc_boot[j], auc_ref)

    return mape_boot, auc_boot, composite_boot


def ci95(arr):
    """Return (lower, upper) 95% percentile bootstrap CI."""
    return np.percentile(arr, 2.5), np.percentile(arr, 97.5)


# =============================================================================
# Reference values (indexed by source: 0=amsterdam, 1=mimic4, 2=eicu+hirid)
# =============================================================================

MAPE_REF = [80.3, 82.8, (87.4 + 83.2) / 2]
AUC_REF  = [0.774, 0.858, (0.788 + 0.851) / 2]

# Paper values indexed by (repo_idx * 3 + source_idx)
# On explicite clairement la structure pour éviter les décalages silencieux
PAPER = {
    #  repo                           amsterdam        mimic4          eicu+hirid
    "amsterdam_25":                 [(97.03, 0.70, 1.41), (95.76, 0.77, 1.32), (117.08, 0.78, 1.20)],
    "mimic4_75":                    [(105.53, 0.71, 1.43), (79.99, 0.86, 0.94), (127.29, 0.78, 1.25)],
    "amsterdam_25_retrain_all_layers": [(88.04, 0.76, 1.21), (np.nan, np.nan, np.nan), (np.nan, np.nan, np.nan)],
    "amsterdam_mimic4_100":         [(86.0, 0.74, 1.26), (80.47, 0.84, 1.02), (117.02, 0.82, 1.09)],
}

SOURCES = ["amsterdam", "mimic4", ["eicu", "hirid"]]
REPOS   = list(PAPER.keys())


# =============================================================================
# Main loop
# =============================================================================

for repo in REPOS:
    print(f"\n{'='*60}")
    print(f"Training on: {repo}")
    print(f"{'='*60}")

    df_all = load_predictions(repo)
    composite_gaps = []

    for k, source in enumerate(SOURCES):
        mape_paper, auc_paper, composite_paper = PAPER[repo][k]
        mape_ref = MAPE_REF[k]
        auc_ref  = AUC_REF[k]

        source_label = source if isinstance(source, str) else str(source)
        print(f"\n---> Evaluating on {source_label}")

        # Filter source + minimum LOS
        df = filter_by_source(df_all.copy(), source)
        df = df[df["label_los"] > 2 / 24]

        # MAPE sur patients survivants uniquement, AUC sur tous
        df_alive = df[df["label_mort"] == 0]

        # Point estimates
        mape         = calculate_mape(df_alive["label_los"], df_alive["pred_los"])
        auc, auc_cov = delong_roc_variance(df["label_mort"].to_numpy(), df["pred_mort"].to_numpy())
        auc_std      = np.sqrt(auc_cov)
        composite    = calculate_composite(mape, mape_ref, auc, auc_ref)

        print(f"MAPE      : {mape:.2f}  (paper: {mape_paper:.2f})")
        print(f"AUC       : {auc:.2f}  (paper: {auc_paper:.2f})")
        print(f"Composite : {composite:.2f}  (paper: {composite_paper:.2f})")
        print(f"  gaps -> MAPE: {abs(mape - mape_paper):.2f}%, "
              f"AUC: {abs(auc - auc_paper):.2f}, "
              f"Composite: {abs(composite - composite_paper):.2f}")

        # Bootstrap CI
        mape_boot, auc_boot, composite_boot = bootstrap_metrics(df, df_alive, mape_ref, auc_ref)

        mape_lo, mape_hi           = ci95(mape_boot)
        composite_lo, composite_hi = ci95(composite_boot)

        print(f"  IC95% MAPE      : [{mape_lo:.3f}%, {mape_hi:.3f}%]")
        print(f"  IC95% Composite : [{composite_lo:.3f}, {composite_hi:.3f}]")

        # AUC CI via DeLong
        auc_lo = auc - 1.96 * auc_std
        auc_hi = auc + 1.96 * auc_std
        print(f"  IC95% AUC       : [{auc_lo:.3f}, {auc_hi:.3f}]  (DeLong)")

        # On ne transfère plus l'écart bootstrap vers les valeurs papier
        # car la variance dépend du dataset original
        if not np.isnan(mape_paper):
            composite_gaps.append(abs(composite - composite_paper))

    if composite_gaps:
        print(f"\nMean composite gap across sources: {np.mean(composite_gaps):.4f}")
    else:
        print("\nMean composite gap: N/A (missing paper values)")