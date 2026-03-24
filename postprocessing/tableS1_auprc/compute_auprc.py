
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt


def calculate_auprc(y_true, y_pred):
    if y_true.nunique() == 1:
        return np.nan
    return average_precision_score(y_true, y_pred)


# =============================================================================
# ## Table S1 : AUPRC computation + Precision-Recall curves
# =============================================================================
source_labels = {"amsterdam": "AmsterdamUMC", "eicu": "eICU",
                 "hirid": "HiRID", "mimic4": "MIMIC-IV"}

fig, axes = plt.subplots(3, 4, figsize=(16, 12))

for row, model in enumerate(["LSTM", "Transformer", "TPC"]):
    if model == "TPC":
        path = ("Y:/DDS_Rocheteau/BlendedLOS/results/model_benchmark_ok/"
                 "amsterdam_eicu_hirid_mimic4_20/multitask/" + model
                 + "/24-04-17_104916/test_predictions_mort.csv")
    else:
        path = ("Y:/DDS_Rocheteau/BlendedLOS/results/model_benchmark_ok/"
                 "amsterdam_eicu_hirid_mimic4_20/multitask/" + model
                 + "/24-04-17_104917/test_predictions_mort.csv")

    df_pred = pd.read_csv(path)
    df_pred.rename(columns={"label": "label_mort"}, inplace=True)
    print(f"current model is {model}")

    for col, source in enumerate(["amsterdam", "eicu", "hirid", "mimic4"]):
        # Validation
        df_source = df_pred[df_pred['patientids'].str.contains(source, case=False, na=False)]

        # # Patient level agregation
        # df_source = df_source.groupby('patientids').last().reset_index()

        y_true_mort  = df_source["label_mort"]
        y_score_mort = df_source["pred_mort"]

        # Point estimate
        auprc = calculate_auprc(y_true_mort, y_score_mort)
        print(f"evaluating on {source}")
        print(f"AUPRC : {auprc:.3f}")

        # Bootstrap IC95%
        n_bootstrap = 10000
        auprc_bootstrapped = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            y_true_r, y_score_r = resample(y_true_mort, y_score_mort, random_state=i)
            auprc_bootstrapped[i] = calculate_auprc(
                pd.Series(y_true_r),
                pd.Series(y_score_r)
            )
        lower_bound = np.percentile(auprc_bootstrapped, 2.5)
        upper_bound = np.percentile(auprc_bootstrapped, 97.5)
        print(f"IC95% AUPRC (Bootstrap) : [{lower_bound:.3f}, {upper_bound:.3f}]")

        # Precision-Recall curve
        baseline = y_true_mort.mean()
        print(f"AUPRC baseline (prevalence) : {baseline:.3f}")
        precision, recall, _ = precision_recall_curve(y_true_mort, y_score_mort)

        ax = axes[row][col]
        ax.plot(recall, precision, color='steelblue', lw=1.5,
                label=f'AUPRC = {auprc:.3f}')
        ax.axhline(y=baseline, color='gray', linestyle='--', lw=1,
                   label=f'Baseline = {baseline:.3f}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'{model} — {source_labels[source]}')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(loc='upper right', fontsize=8)

    print("\n")

plt.suptitle('Precision-Recall curves — Model benchmark (int val)', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('pr_curves_model_benchmark_patient_agregation.png', dpi=150, bbox_inches='tight')
plt.show()