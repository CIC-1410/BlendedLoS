import pandas as pd
import numpy as np
import re
from sklearn import metrics
from sklearn.utils import resample
from rocauc_comparison import delong_roc_variance

def calculate_mape(y_true, y_pred):
    return ((y_true - y_pred).abs() / y_true).mean() * 100

def calculat_compo(mape, mape_ref, auc, auc_ref):
    return 0.5 * (mape / mape_ref + (1 - auc) / (1 - auc_ref))
# =============================================================================
## table 3: IC computation
# =============================================================================
i=0
for repo in ["amsterdam_25", "mimic4_75", "amsterdam_mimic4_100"]:
    print("training on", repo)
    k=0
    mean_gap = []
    for source in ["amsterdam", "mimic4", ["eicu","hirid"]]:
        path1 = "Z:/DDS_Rocheteau/BlendedLOS/results_mimic4update-server/main_experiment/"+repo+"/multitask/TPC/26-03-20_212503/test_predictions_los.csv"
        path2 = "Z:/DDS_Rocheteau/BlendedLOS/results_mimic4update-server/main_experiment/"+repo+"/multitask/TPC/26-03-20_212503/test_predictions_mort.csv"
        df_los = pd.read_csv(path1)
        df_pred = pd.read_csv(path2)
        df_pred["pred_los"], df_pred["label_los"] = df_los["pred_los"],df_los["label"]
        df_pred.rename(columns={"label":"label_mort"}, inplace=True)
        del path1, path2, df_los
        val_mode = "int_val"
        if type(source) == str:
            df_pred = df_pred[df_pred['patientids'].str.contains(source, case=False, na=False)]
        else:
            pattern = '|'.join(map(re.escape, source))
            df_pred = df_pred[df_pred['patientids'].str.contains(pattern, case=False, na=False)]
        
        # df_pred = df_pred[df_pred["label_mort"]==0]
        df_pred = df_pred[df_pred["label_los"]>2/24]
        # y_true, y_pred = df_pred["label_los"], df_pred["pred_los"]
        
        mape = calculate_mape(df_pred["label_los"], df_pred["pred_los"])
        ground_truth = df_pred["label_mort"].to_numpy()
        predictions = df_pred["pred_mort"].to_numpy()
        auc, _ = delong_roc_variance(ground_truth, predictions)
        mape_ref = [80.3, 82.8, (87.4+83.2)/2]
        auc_ref = [0.774, 0.858, (0.788+0.851)/2]
        mape_paper = [97.03,95.76,117.08, 105.53,79.99,127.29,88.04,np.nan,np.nan,86,80.47,117.02]
        auc_paper = [0.70, 0.77, 0.78, 0.71, 0.86, 0.78, 0.76, np.nan,np.nan,0.74,0.84,0.82]
        composite_paper = [1.41,1.32,1.20,1.43,0.94,1.25,1.21,np.nan,np.nan,1.26,1.02,1.09]
        print(f"---> evaluating on {source}")
        print(f"MAPE observée : {mape:.3f}%; MAPE reelle: {mape_paper[i]}")
        print(f"AUC observée : {auc:.3f}%; AUC reelle: {auc_paper[i]}")
        
        composite = calculat_compo(mape, mape_ref[k], auc, auc_ref[k])
        print(f"Met composite observée : {composite:.3f}; Met composite reelle: {composite_paper[i]}")
       
        print("MAPE gap w paper:", abs(mape-mape_paper[i]),"%")
        print("gap w paper:", abs(auc-auc_paper[i]))
        print("composite gap w paper:", abs(composite-composite_paper[i]))
        mean_gap.append(abs(composite-composite_paper[i]))
        
        
        ## Bootstrap pour estimer l'IC95% de la MAPE
        n_bootstrap = 1000  # Nombre de rééchantillonnages
        mape_bootstrapped = np.zeros(n_bootstrap)
        auc_bootstrapped = np.zeros(n_bootstrap)
        composite_bootstrapped = np.zeros(n_bootstrap)
        for j in range(n_bootstrap):
            y_mort_true, y_mort_pred = resample(df_pred["label_mort"], df_pred["pred_mort"], random_state=j)
            y_los_true, y_los_pred = resample(df_pred["label_los"], df_pred["pred_los"], random_state=j)
            # df_pred_resampled = resample(df_pred, random_state=j)
            # auc_bootstrapped[j], _ = delong_roc_variance(y_mort_true.to_numpy(), y_mort_pred.to_numpy())
            fpr, tpr, _ = metrics.roc_curve(y_mort_true, y_mort_pred, pos_label=1)
            auc_bootstrapped[j] = metrics.auc(fpr, tpr)
            mape_bootstrapped[j] = calculate_mape(y_los_true, y_los_pred)
            composite_bootstrapped[j] = calculat_compo(mape_bootstrapped[j], mape_ref[k], auc_bootstrapped[j], auc_ref[k])
        
        ## Calcul de l'IC95%
        lower_bound = np.percentile(mape_bootstrapped, 2.5)
        upper_bound = np.percentile(mape_bootstrapped, 97.5)
        print(f"IC95% pour la MAPE : [{lower_bound:.3f}%, {upper_bound:.3f}%]")
        ecart_1, ecart_2 = mape-lower_bound, upper_bound-mape
        print(f"IC95% pour la MAPE reelle : [{mape_paper[k]-ecart_1} - {mape_paper[k]+ecart_2}]")
        lower_bound = np.percentile(composite_bootstrapped, 2.5)
        upper_bound = np.percentile(composite_bootstrapped, 97.5)
        print(f"IC95% pour la composite : [{lower_bound:.3f}%, {upper_bound:.3f}%]")
        ecart_1, ecart_2 = composite-lower_bound, upper_bound-composite
        print(f"IC95% pour la composite reelle : [{composite_paper[i]-ecart_1} - {composite_paper[i]+ecart_2}]")
       
        k+=1
        i+=1
        
        # compute auc
        y_true, y_score = df_pred["label_mort"], df_pred["pred_mort"]
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
        print("AUROC mortality int val:", metrics.auc(fpr, tpr))
        # compute delong variance for AUC
        ground_truth = y_true.to_numpy()
        predictions = y_score.to_numpy()
        auc, auc_cov = delong_roc_variance(ground_truth, predictions)
        auc_std = np.sqrt(auc_cov)  # Écart-type de l'AUC
        lower_bound = auc - 1.96 * auc_std
        upper_bound = auc + 1.96 * auc_std
        print(f"AUC : {auc:.3f}")
        print(f"IC95% pour l'AUC : [{lower_bound}, {upper_bound}]")
        print(f"IC95% pour l'AUC : [{lower_bound:.3f}, {upper_bound:.3f}]")
        
    print("mean composite gap:",np.mean(mean_gap),"%") # gap accros evaluation on 
    print("\n")