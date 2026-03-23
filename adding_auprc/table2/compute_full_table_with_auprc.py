import pandas as pd
import numpy as np
import re
from sklearn import metrics
from sklearn.utils import resample
from sklearn.metrics import average_precision_score
from rocauc_comparison import delong_roc_variance

def calculate_mape(y_true, y_pred):
    return ((y_true - y_pred).abs() / y_true).mean() * 100

def calculate_auprc(y_true, y_pred):
    if y_true.nunique() == 1:
        return np.nan
    return average_precision_score(y_true, y_pred)

def calculat_compo(mape, mape_ref, auc, auc_ref):
    return 0.5*(mape/mape_ref +(1-auc)/(1-auc_ref))

# =============================================================================
# ## table 2: IC95% computation via Bootstrap+Delong
# =============================================================================
val_mode = "int_val"
# val_mode = "ext_val"
k = 0
mean_gap = []
for source in ["mimic4"]:
    print(f'{source}')
    path1 = "Z:/DDS_Rocheteau/BlendedLOS/results_mimic4update-server/dataset_benchmark/"+source+"_75/multitask/TPC/26-03-20_212503/test_predictions_los.csv"
    path2 = "Z:/DDS_Rocheteau/BlendedLOS/results_mimic4update-server/dataset_benchmark/"+source+"_75/multitask/TPC/26-03-20_212503/test_predictions_mort.csv"
    df_los = pd.read_csv(path1)
    df_pred = pd.read_csv(path2)
    df_pred["pred_los"], df_pred["label_los"] = df_los["pred_los"], df_los["label"]
    df_pred.rename(columns={"label": "label_mort"}, inplace=True)
    del path1, path2, df_los

    if val_mode == "int_val":
        df_pred = df_pred[df_pred['patientids'].str.contains(source, case=False, na=False)]
        mape_paper = [80.3, 87.4, 83.2, 82.8]  # int val
        auc_paper  = [0.774, 0.788, 0.851, 0.858]  # int val
    else:
        df_pred = df_pred[~df_pred['patientids'].str.contains(source, case=False, na=False)]
        mape_paper = [91.4, 98.0, 142.5, 83.2]  # ext val
        auc_paper  = [0.738, 0.737, 0.801, 0.771]  # ext val

    # Agregation by patient
    ## df_pred = df_pred.groupby('patientids').last().reset_index()

    # -------------------------------------------------------------------------
    ## compute rlos mape — filtre: survivants uniquement + séjours > 2h
    # -------------------------------------------------------------------------
    df_rlos = df_pred[df_pred["label_mort"] == 0]
    df_rlos = df_rlos[df_rlos["label_los"] > 2/24]
    y_true_los, y_pred_los = df_rlos["label_los"], df_rlos["pred_los"]

    mape = calculate_mape(y_true_los, y_pred_los)
    print(f'MAPE los {val_mode}:', round(mape, 3), "% mape in paper:", mape_paper[k], '%')
    print("gap w paper:", abs(mape - mape_paper[k]))
    mean_gap.append(abs(mape - mape_paper[k]))

    ## Bootstrap pour IC95% de la MAPE
    n_bootstrap = 10000
    mape_bootstrapped = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        y_true_resampled, y_pred_resampled = resample(y_true_los, y_pred_los, random_state=i)
        mape_bootstrapped[i] = calculate_mape(y_true_resampled, y_pred_resampled)

    lower_bound = np.percentile(mape_bootstrapped, 2.5)
    upper_bound = np.percentile(mape_bootstrapped, 97.5)
    print(f"IC95% pour la MAPE {val_mode}: [{lower_bound:.3f}%, {upper_bound:.3f}%]")
    ecart_1, ecart_2 = mape - lower_bound, upper_bound - mape
    print(f"IC95% pour la MAPE reelle : [{mape_paper[k]-ecart_1:.3f} - {mape_paper[k]+ecart_2:.3f}]")

    # -------------------------------------------------------------------------
    ## compute mortality AUROC via DeLong
    # -------------------------------------------------------------------------
    y_true_mort, y_score_mort = df_pred["label_mort"], df_pred["pred_mort"]

    ground_truth = y_true_mort.to_numpy()
    predictions  = y_score_mort.to_numpy()
    auc, auc_cov = delong_roc_variance(ground_truth, predictions)
    print(f'delong {val_mode} AUC variance is: {auc_cov}')
    auc_std     = np.sqrt(auc_cov)
    lower_bound = auc - 1.96 * auc_std
    upper_bound = auc + 1.96 * auc_std
    print(f"AUC : {auc:.3f}", "in paper:", auc_paper[k])
    print(f"IC95% pour l'AUC (DeLong) : [{lower_bound:.3f}, {upper_bound:.3f}]")
    ecart_1, ecart_2 = auc - lower_bound, upper_bound - auc
    print(f"IC95% pour l'AUC reelle : [{auc_paper[k]-ecart_1:.3f} - {auc_paper[k]+ecart_2:.3f}]")

    # -------------------------------------------------------------------------
    ## compute mortality AUPRC via Bootstrap (pas d'équivalent DeLong )
    # -------------------------------------------------------------------------
    auprc = calculate_auprc(y_true_mort, y_score_mort)
    print(f"AUPRC : {auprc:.3f}")

    auprc_bootstrapped = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        y_true_resampled, y_score_resampled = resample(y_true_mort, y_score_mort, random_state=i)
        auprc_bootstrapped[i] = calculate_auprc(
            pd.Series(y_true_resampled),
            pd.Series(y_score_resampled)
        )

    lower_bound = np.percentile(auprc_bootstrapped, 2.5)
    upper_bound = np.percentile(auprc_bootstrapped, 97.5)
    print(f"IC95% pour l'AUPRC (Bootstrap) : [{lower_bound:.3f}, {upper_bound:.3f}]")

    print("\n")
    k += 1
    print("mean gap:", np.mean(mean_gap))

# -------------------------------------------------------------------------
# ## Precision-Recall curves plot for each dataset
# -------------------------------------------------------------------------
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()
sources = ["amsterdam", "eicu", "hirid", "mimic4"]
source_labels = {"amsterdam": "AmsterdamUMC", "eicu": "eICU",
                 "hirid": "HiRID", "mimic4": "MIMIC-IV"}

for k, source in enumerate(sources):
    path2 = ("Z:/DDS_Rocheteau/BlendedLOS/results_mimic4update-server/dataset_benchmark/"
             + source + "_75/multitask/TPC/26-03-20_212503/test_predictions_mort.csv")
    df = pd.read_csv(path2)
    df.rename(columns={"label": "label_mort"}, inplace=True)

    # Filter internal or external 
    if val_mode == "int_val":
        df = df[df['patientids'].str.contains(source, case=False, na=False)]
    else:
        df = df[~df['patientids'].str.contains(source, case=False, na=False)]

    # Agregation by patient
    ## df = df.groupby('patientids').last().reset_index()

    y_true  = df["label_mort"]
    y_score = df["pred_mort"]

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auprc = calculate_auprc(y_true, y_score)

    # Baseline = prevalence of positive class (random classifier)
    baseline = y_true.mean()

    ax = axes[k]
    ax.plot(recall, precision, color='steelblue', lw=1.5,
            label=f'AUPRC = {auprc:.3f}')
    ax.axhline(y=baseline, color='gray', linestyle='--', lw=1,
               label=f'Baseline = {baseline:.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'{source_labels[source]} ({val_mode})')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc='upper right', fontsize=9)

plt.suptitle('Precision-Recall curves — TPC model', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'pr_curves_{val_mode}.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Courbes PR sauvegardées dans pr_curves_{val_mode}.png")