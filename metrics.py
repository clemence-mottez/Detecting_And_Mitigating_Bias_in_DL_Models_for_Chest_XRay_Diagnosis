import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from scipy.stats import entropy



def compute_metrics(predictions, targets, diseases):
    # Calculate metrics for each disease
    metrics = {}
    for idx, disease in enumerate(diseases):
        disease_pred = predictions[disease]
        disease_true = targets[disease]
        # disease_pred = predictions[:, idx]
        # disease_true = targets[:, idx]
        auc_roc = roc_auc_score(disease_true, disease_pred)
        f1 = f1_score(disease_true, (disease_pred > 0.5).astype(int))
        accuracy = accuracy_score(disease_true, (disease_pred > 0.5).astype(int))
        tn, fp, fn, tp = confusion_matrix(disease_true, (disease_pred > 0.5).astype(int)).ravel()
        tp_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        tn_rate = tn / (tn + fp) if (tn + fp) > 0 else 0
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        fp_rate = fp / (tn + fp) if (tn + fp) > 0 else 0
        
        # Calculate Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(disease_true, disease_pred)
        auprc = auc(recall, precision)

        metrics[disease] = {
            'Accuracy': accuracy,
            'AUC': auc_roc,
            'AUPRC': auprc,
            'F1 Score': f1,
            'TP Rate': tp_rate,
            'FN Rate': fn_rate,
            'TN Rate': tn_rate,
            'FP Rate': fp_rate
        }
    return metrics



def compute_kl_divergence_sex(y_test, y_pred, y_sex):
    results = {}
    
    
    # Calculating the KL-Divergence for every disease
    for disease, y_pred_probs in y_pred.items():
        # True Labels for the conditions
        y_test_labels = y_test[disease]
        
        # Male Subgruppe
        y_true_male = y_test_labels[y_sex == 1] 
        y_pred_male = np.array(y_pred_probs)[y_sex == 1] 
        
        # Female Subgruppe
        y_true_female = y_test_labels[y_sex == 0] 
        y_pred_female = np.array(y_pred_probs)[y_sex == 0] 
        
        # Function to create the probability distribution
        def create_distributions(y_true, y_pred):
            P = np.array([1 - y_true, y_true]).T  # Probabiility distribution of True Labels
            Q = np.array([1 - y_pred, y_pred]).T  # Probabiility distribution of predicted diseases
            return P, Q
        
        # Male KL-Divergence
        P_male, Q_male = create_distributions(y_true_male, y_pred_male)
        kl_div_male = np.mean([entropy(P_row, Q_row) for P_row, Q_row in zip(P_male, Q_male)])
        
        # Female KL-Divergence
        P_female, Q_female = create_distributions(y_true_female, y_pred_female)
        kl_div_female = np.mean([entropy(P_row, Q_row) for P_row, Q_row in zip(P_female, Q_female)])
        
        # Save the results
        results[disease] = {
            'male': kl_div_male,
            'female': kl_div_female
        }
    
    return results



def compute_kl_divergence_age(y_test, y_pred, y_age):

    results = {}

    # Calculating the KL-Divergence for every disease
    for disease, y_pred_probs in y_pred.items():
        # True Labels for the conditions
        y_test_labels = y_test[disease]
        
        # Over 70
        y_true_over = y_test_labels[y_age == 1] 
        y_pred_over = np.array(y_pred_probs)[y_age == 1] 
        
        # Under 70
        y_true_under = y_test_labels[y_age == 0] 
        y_pred_under = np.array(y_pred_probs)[y_age == 0] 
        
        # Function to create the probability distribution
        def create_distributions(y_true, y_pred):
            P = np.array([1 - y_true, y_true]).T  # Probabiility distribution of True Labels
            Q = np.array([1 - y_pred, y_pred]).T  # Probabiility distribution of predicted diseases
            return P, Q
        
        # Over 70 KL-Divergence
        P_over, Q_over = create_distributions(y_true_over, y_pred_over)
        kl_div_over = np.mean([entropy(P_row, Q_row) for P_row, Q_row in zip(P_over,Q_over)])
        
        # Under 70 KL-Divergence
        P_under, Q_under = create_distributions(y_true_under, y_pred_under)
        kl_div_under = np.mean([entropy(P_row, Q_row) for P_row, Q_row in zip(P_under,Q_under)])
                                
        # Save the results
        results[disease] = {
            'over_70': kl_div_over,
            'under_70': kl_div_under
        }
    
    return results


def compute_kl_divergence_race(y_test, y_pred, y_race):
    results = {}
    
    
    # Calculating the KL-Divergence for every disease
    for disease, y_pred_probs in y_pred.items():
        # True Labels for the conditions
        y_test_labels = y_test[disease]
        
        # White Subgroup
        y_true_white = y_test_labels[y_race == 0] 
        y_pred_white = np.array(y_pred_probs)[y_race == 0] 
        
        # Black subgroup
        y_true_black = y_test_labels[y_race == 2] 
        y_pred_black = np.array(y_pred_probs)[y_race == 2] 

        # Asian subgroup
        y_true_asian = y_test_labels[y_race == 1] 
        y_pred_asian = np.array(y_pred_probs)[y_race == 1] 
        
        # Function to create the probability distribution
        def create_distributions(y_true, y_pred):
            P = np.array([1 - y_true, y_true]).T  # Probabiility distribution of True Labels
            Q = np.array([1 - y_pred, y_pred]).T  # Probabiility distribution of predicted diseases
            return P, Q
        
        # White KL-Divergence
        P_white, Q_white = create_distributions(y_true_white, y_pred_white)
        kl_div_white= np.mean([entropy(P_row, Q_row) for P_row, Q_row in zip(P_white, Q_white)])
        
        # Black KL-Divergence
        P_black, Q_black = create_distributions(y_true_black, y_pred_black)
        kl_div_black = np.mean([entropy(P_row, Q_row) for P_row, Q_row in zip(P_black, Q_black)])

        # Asian KL-Divergence
        P_asian, Q_asian = create_distributions(y_true_asian, y_pred_asian)
        kl_div_asian = np.mean([entropy(P_row, Q_row) for P_row, Q_row in zip(P_asian, Q_asian)])
        
        # Save the results
        results[disease] = {
            'white': kl_div_white,
            'black': kl_div_black,
            'asian': kl_div_asian
        }
    
    return results



def compute_metrics_subcath(predictions, targets, diseases, y_sex, y_race, y_age, race = False): #kl_divergence_results_age, kl_divergence_results_race, kl_divergence_results_sex, race = False):
    
    # Calculate metrics for each disease and for each class

    metrics_female = {}
    for idx, disease in enumerate(diseases):
        # disease_pred = predictions[:, idx]
        disease_pred = predictions[y_sex == 1, idx]
        disease_true = targets[y_sex == 1, idx]
        auc_roc = roc_auc_score(disease_true, disease_pred)
        f1 = f1_score(disease_true, (disease_pred > 0.5).astype(int))
        accuracy = accuracy_score(disease_true, (disease_pred > 0.5).astype(int))
        tn, fp, fn, tp = confusion_matrix(disease_true, (disease_pred > 0.5).astype(int)).ravel()
        tp_rate = tp / (tp + fn)
        tn_rate = tn / (tn + fp)
        fn_rate = fn / (fn + tp)
        fp_rate = fp / (tn + fp)
        
        precision, recall, _ = precision_recall_curve(disease_true, disease_pred)
        auprc = auc(recall, precision)


        metrics_female[disease] = {
            'Accuracy': accuracy,
            'AUC': auc_roc,
            'AUPRC': auprc,
            'F1 Score': f1,
            'TP Rate': tp_rate,
            'FN Rate': fn_rate,
            'TN Rate': tn_rate,
            'FP Rate': fp_rate,
            # 'KL div': kl_divergence_results_sex[disease]['female']
            }
        
    metrics_male = {}
    for idx, disease in enumerate(diseases):
        # disease_pred = predictions[:, idx]
        disease_pred = predictions[y_sex == 0, idx]
        disease_true = targets[y_sex == 0, idx]
        auc_roc = roc_auc_score(disease_true, disease_pred)
        f1 = f1_score(disease_true, (disease_pred > 0.5).astype(int))
        accuracy = accuracy_score(disease_true, (disease_pred > 0.5).astype(int))
        tn, fp, fn, tp = confusion_matrix(disease_true, (disease_pred > 0.5).astype(int)).ravel()
        tp_rate = tp / (tp + fn)
        tn_rate = tn / (tn + fp)
        fn_rate = fn / (fn + tp)
        fp_rate = fp / (tn + fp)

        precision, recall, _ = precision_recall_curve(disease_true, disease_pred)
        auprc = auc(recall, precision)


        metrics_male[disease] = {
            'Accuracy': accuracy,
            'AUC': auc_roc,
            'AUPRC': auprc,
            'F1 Score': f1,
            'TP Rate': tp_rate,
            'FN Rate': fn_rate,
            'TN Rate': tn_rate,
            'FP Rate': fp_rate,
            # 'KL div': kl_divergence_results_sex[disease]['male']
            }
        
    if race:
        metrics_white = {}
        for idx, disease in enumerate(diseases):
            # disease_pred = predictions[:, idx]
            disease_pred = predictions[y_race == 0, idx]
            disease_true = targets[y_race == 0, idx]
            auc_roc = roc_auc_score(disease_true, disease_pred)
            f1 = f1_score(disease_true, (disease_pred > 0.5).astype(int))
            accuracy = accuracy_score(disease_true, (disease_pred > 0.5).astype(int))
            tn, fp, fn, tp = confusion_matrix(disease_true, (disease_pred > 0.5).astype(int)).ravel()
            tp_rate = tp / (tp + fn)
            tn_rate = tn / (tn + fp)
            fn_rate = fn / (fn + tp)
            fp_rate = fp / (tn + fp)

            precision, recall, _ = precision_recall_curve(disease_true, disease_pred)
            auprc = auc(recall, precision)


            metrics_white[disease] = {
                'Accuracy': accuracy,
                'AUC': auc_roc,
                'AUPRC': auprc,
                'F1 Score': f1,
                'TP Rate': tp_rate,
                'FN Rate': fn_rate,
                'TN Rate': tn_rate,
                'FP Rate': fp_rate,
                # 'KL div': kl_divergence_results_race[disease]['white']
                }
            
        metrics_black = {}
    
        metrics_asian = {}
        for idx, disease in enumerate(diseases):
            # disease_pred = predictions[:, idx]
            disease_pred = predictions[y_race == 1, idx]
            disease_true = targets[y_race == 1, idx]
            auc_roc = roc_auc_score(disease_true, disease_pred)
            f1 = f1_score(disease_true, (disease_pred > 0.5).astype(int))
            accuracy = accuracy_score(disease_true, (disease_pred > 0.5).astype(int))
            tn, fp, fn, tp = confusion_matrix(disease_true, (disease_pred > 0.5).astype(int)).ravel()
            tp_rate = tp / (tp + fn)
            tn_rate = tn / (tn + fp)
            fn_rate = fn / (fn + tp)
            fp_rate = fp / (tn + fp)

            precision, recall, _ = precision_recall_curve(disease_true, disease_pred)
            auprc = auc(recall, precision)


            metrics_asian[disease] = {
                'Accuracy': accuracy,
                'AUC': auc_roc,
                'AUPRC': auprc,
                'F1 Score': f1,
                'TP Rate': tp_rate,
                'FN Rate': fn_rate,
                'TN Rate': tn_rate,
                'FP Rate': fp_rate,
                # 'KL div': kl_divergence_results_race[disease]['asian']
                }
            
            


    else:
        metrics_white = {}
        for idx, disease in enumerate(diseases):
            # disease_pred = predictions[:, idx]
            disease_pred = predictions[y_race == 0, idx]
            disease_true = targets[y_race == 0, idx]
            auc_roc = roc_auc_score(disease_true, disease_pred)
            f1 = f1_score(disease_true, (disease_pred > 0.5).astype(int))
            accuracy = accuracy_score(disease_true, (disease_pred > 0.5).astype(int))
            tn, fp, fn, tp = confusion_matrix(disease_true, (disease_pred > 0.5).astype(int)).ravel()
            tp_rate = tp / (tp + fn)
            tn_rate = tn / (tn + fp)
            fn_rate = fn / (fn + tp)
            fp_rate = fp / (tn + fp)

            precision, recall, _ = precision_recall_curve(disease_true, disease_pred)
            auprc = auc(recall, precision)


            metrics_white[disease] = {
                'Accuracy': accuracy,
                'AUC': auc_roc,
                'AUPRC': auprc,
                'F1 Score': f1,
                'TP Rate': tp_rate,
                'FN Rate': fn_rate,
                'TN Rate': tn_rate,
                'FP Rate': fp_rate,
                # 'KL div': kl_divergence_results_race[disease]['white']
                }
            
        metrics_black = {}
        for idx, disease in enumerate(diseases):
            # disease_pred = predictions[:, idx]
            disease_pred = predictions[y_race == 2, idx]
            disease_true = targets[y_race == 2, idx]
            auc_roc = roc_auc_score(disease_true, disease_pred)
            f1 = f1_score(disease_true, (disease_pred > 0.5).astype(int))
            accuracy = accuracy_score(disease_true, (disease_pred > 0.5).astype(int))
            tn, fp, fn, tp = confusion_matrix(disease_true, (disease_pred > 0.5).astype(int)).ravel()
            tp_rate = tp / (tp + fn)
            tn_rate = tn / (tn + fp)
            fn_rate = fn / (fn + tp)
            fp_rate = fp / (tn + fp)

            precision, recall, _ = precision_recall_curve(disease_true, disease_pred)
            auprc = auc(recall, precision)


            metrics_black[disease] = {
                'Accuracy': accuracy,
                'AUC': auc_roc,
                'AUPRC': auprc,
                'F1 Score': f1,
                'TP Rate': tp_rate,
                'FN Rate': fn_rate,
                'TN Rate': tn_rate,
                'FP Rate': fp_rate,
                # 'KL div': kl_divergence_results_race[disease]['black']
                }
            
        metrics_asian = {}
        for idx, disease in enumerate(diseases):
            # disease_pred = predictions[:, idx]
            disease_pred = predictions[y_race == 1, idx]
            disease_true = targets[y_race == 1, idx]
            auc_roc = roc_auc_score(disease_true, disease_pred)
            f1 = f1_score(disease_true, (disease_pred > 0.5).astype(int))
            accuracy = accuracy_score(disease_true, (disease_pred > 0.5).astype(int))
            tn, fp, fn, tp = confusion_matrix(disease_true, (disease_pred > 0.5).astype(int)).ravel()
            tp_rate = tp / (tp + fn)
            tn_rate = tn / (tn + fp)
            fn_rate = fn / (fn + tp)
            fp_rate = fp / (tn + fp)

            precision, recall, _ = precision_recall_curve(disease_true, disease_pred)
            auprc = auc(recall, precision)


            metrics_asian[disease] = {
                'Accuracy': accuracy,
                'AUC': auc_roc,
                'AUPRC': auprc,
                'F1 Score': f1,
                'TP Rate': tp_rate,
                'FN Rate': fn_rate,
                'TN Rate': tn_rate,
                'FP Rate': fp_rate,
                # 'KL div': kl_divergence_results_race[disease]['asian']
                }
            
            

    metrics_young = {}
    for idx, disease in enumerate(diseases):
        # disease_pred = predictions[:, idx]
        disease_pred = predictions[y_age == 0, idx]
        disease_true = targets[y_age == 0, idx]
        auc_roc = roc_auc_score(disease_true, disease_pred)
        f1 = f1_score(disease_true, (disease_pred > 0.5).astype(int))
        accuracy = accuracy_score(disease_true, (disease_pred > 0.5).astype(int))
        tn, fp, fn, tp = confusion_matrix(disease_true, (disease_pred > 0.5).astype(int)).ravel()
        tp_rate = tp / (tp + fn)
        tn_rate = tn / (tn + fp)
        fn_rate = fn / (fn + tp)
        fp_rate = fp / (tn + fp)
        
        precision, recall, _ = precision_recall_curve(disease_true, disease_pred)
        auprc = auc(recall, precision)


        metrics_young[disease] = {
            'Accuracy': accuracy,
            'AUC': auc_roc,
            'AUPRC': auprc,
            'F1 Score': f1,
            'TP Rate': tp_rate,
            'FN Rate': fn_rate,
            'TN Rate': tn_rate,
            'FP Rate': fp_rate,
            # 'KL div': kl_divergence_results_age[disease]['under_70']
            }
        
    metrics_old = {}
    for idx, disease in enumerate(diseases):
        # disease_pred = predictions[:, idx]
        disease_pred = predictions[y_age == 1, idx]
        disease_true = targets[y_age == 1, idx]
        auc_roc = roc_auc_score(disease_true, disease_pred)
        f1 = f1_score(disease_true, (disease_pred > 0.5).astype(int))
        accuracy = accuracy_score(disease_true, (disease_pred > 0.5).astype(int))
        tn, fp, fn, tp = confusion_matrix(disease_true, (disease_pred > 0.5).astype(int)).ravel()
        tp_rate = tp / (tp + fn)
        tn_rate = tn / (tn + fp)
        fn_rate = fn / (fn + tp)
        fp_rate = fp / (tn + fp)

        precision, recall, _ = precision_recall_curve(disease_true, disease_pred)
        auprc = auc(recall, precision)


        metrics_old[disease] = {
            'Accuracy': accuracy,
            'AUC': auc_roc,
            'AUPRC': auprc,
            'F1 Score': f1,
            'TP Rate': tp_rate,
            'FN Rate': fn_rate,
            'TN Rate': tn_rate,
            'FP Rate': fp_rate,
            # 'KL div': kl_divergence_results_age[disease]['over_70']

            }
    
    return metrics_female, metrics_male, metrics_white, metrics_black, metrics_asian, metrics_young, metrics_old




def bias_table(metrics, metrics_female, metrics_male, metrics_white, metrics_black, metrics_asian, metrics_young, metrics_old, race = False):

    # Initialize an empty list to store the data
    data_sex = []

    # Iterate over the diseases in the metrics dictionary
    for disease, values in metrics.items():
        # Extract AUC and rates from dictionaries
        
        auprc_overall = values['AUPRC'] *100

        auc_overall = values['AUC'] *100
        auprc_male = metrics_male[disease]['AUPRC'] *100
        auprc_female = metrics_female[disease]['AUPRC'] *100
        tp_rate_male = metrics_male[disease]['TP Rate'] *100
        tp_rate_female = metrics_female[disease]['TP Rate'] *100
        fp_rate_male = metrics_male[disease]['FP Rate'] *100
        fp_rate_female = metrics_female[disease]['FP Rate'] *100
        fn_rate_male = metrics_male[disease]['FN Rate'] *100
        fn_rate_female = metrics_female[disease]['FN Rate'] *100

        # kl1 = metrics_male[disease]['KL div'] *100
        # kl2 = metrics_female[disease]['KL div'] *100

        
        # delta_KL_sex = abs(kl1 - kl2)


        # Calculate delta AUC and equality of odds
        delta_auc_sex = abs(auprc_male - auprc_female) 
        eq_odds_sex = 0.5 * (abs(tp_rate_male - tp_rate_female) + abs(fp_rate_male - fp_rate_female))
        fnr_diff = abs(fn_rate_male - fn_rate_female)
        
        # Append to the data list
        data_sex.append([disease, auprc_overall, auc_overall, auprc_male, auprc_female, delta_auc_sex, eq_odds_sex, fnr_diff])

    # Create a DataFrame
    df_sex = pd.DataFrame(data_sex, columns=['Disease', 'AUPRC', 'AUC', 'AUC_Male', 'AUC_Female', 'Delta AUC', 'EqOdds', 'FNR_diff'])

    if race:
        data_race = []

        # Iterate over the diseases in the metrics dictionary
        for disease, values in metrics.items():
            # Extract AUC and rates from dictionaries
            
            auprc_overall = values['AUPRC'] *100

            auc_overall = values['AUC'] *100
            auprc_male = metrics_white[disease]['AUPRC'] *100
            auprc_female = metrics_asian[disease]['AUPRC'] *100
            tp_rate_male = metrics_white[disease]['TP Rate'] *100
            tp_rate_female = metrics_asian[disease]['TP Rate'] *100
            fp_rate_male = metrics_white[disease]['FP Rate'] *100
            fp_rate_female = metrics_asian[disease]['FP Rate'] *100
            fn_rate_male = metrics_white[disease]['FN Rate'] *100
            fn_rate_female = metrics_asian[disease]['FN Rate'] *100

            # kl1 = metrics_white[disease]['KL div'] *100
            # kl2 = metrics_asian[disease]['KL div'] *100

            
            # delta_KL_sex = abs(kl1 - kl2)


            # Calculate delta AUC and equality of odds
            delta_auc_sex = abs(auprc_male - auprc_female) 
            eq_odds_sex = 0.5 * (abs(tp_rate_male - tp_rate_female) + abs(fp_rate_male - fp_rate_female))
            fnr_diff = abs(fn_rate_male - fn_rate_female)
            
            # Append to the data list
            data_race.append([disease, auprc_overall, auc_overall, auprc_male, auprc_female, delta_auc_sex, eq_odds_sex, fnr_diff])

        # Create a DataFrame
        df_race = pd.DataFrame(data_race, columns=['Disease', 'AUPRC', 'AUC', 'AUC_White', 'AUC_Asian', 'Delta AUC', 'EqOdds', 'FNR_diff'])


    else:
        # Initialize an empty list to store the data
        data_race = []

        # Iterate over the diseases in the metrics dictionary
        for disease, values in metrics.items():
            auprc_overall = values['AUPRC'] *100
            auc_overall = values['AUC'] *100
            auprc_groups = [
                metrics_white[disease]['AUPRC'] *100,
                metrics_black[disease]['AUPRC'] *100,
                metrics_asian[disease]['AUPRC'] *100
            ]
            tp_rates = [
                metrics_white[disease]['TP Rate'] *100,
                metrics_black[disease]['TP Rate'] *100,
                metrics_asian[disease]['TP Rate'] *100
            ]
            fp_rates = [
                metrics_white[disease]['FP Rate'] *100,
                metrics_black[disease]['FP Rate'] *100,
                metrics_asian[disease]['FP Rate'] *100
            ]

            fn_rates = [
                metrics_white[disease]['FN Rate'] *100,
                metrics_black[disease]['FN Rate'] *100,
                metrics_asian[disease]['FN Rate'] *100
            ]

            # kl_rates = [
            #     metrics_white[disease]['KL div'] *100,
            #     metrics_black[disease]['KL div'] *100,
            #     metrics_asian[disease]['KL div'] *100
            # ]

            # delta_kl_race = max(abs(kl_rates[i] - kl_rates[j]) for i in range(len(kl_rates)) for j in range(i + 1, len(kl_rates)))

            # Calculate the maximum delta AUC
            delta_auprc_race = max(abs(auprc_groups[i] - auprc_groups[j]) for i in range(len(auprc_groups)) for j in range(i + 1, len(auprc_groups)))

            # Calculate the maximum equality of odds
            eq_odds_race = max(
                0.5 * (abs(tp_rates[i] - tp_rates[j]) + abs(fp_rates[i] - fp_rates[j]))
                for i in range(len(tp_rates)) for j in range(i + 1, len(tp_rates))
            )

            fnr_diff_race = max(
                abs(fn_rates[i] - fn_rates[j])
                for i in range(len(fn_rates)) for j in range(i + 1, len(fn_rates))
            )

            # Append to the data list
            data_race.append([disease, auprc_overall, auc_overall] + auprc_groups + [delta_auprc_race, eq_odds_race, fnr_diff_race])

        # Create a DataFrame
        columns = ['Disease', 'AUPRC', 'AUC', 'AUC_White', 'AUC_Black', 'AUC_Asian', 'Max Delta AUC', 'Max EqOdds', 'FNR_diff']
        df_race = pd.DataFrame(data_race, columns=columns)


    # Initialize an empty list to store the data
    data_age = []

    # Iterate over the diseases in the metrics dictionary
    for disease, values in metrics.items():
        # Extract AUC and rates from dictionaries
        
        auprc_overall = values['AUPRC'] *100

        auc_overall = values['AUC'] *100
        auc_old = metrics_old[disease]['AUPRC'] *100
        auc_young = metrics_young[disease]['AUPRC'] *100
        tp_rate_old = metrics_old[disease]['TP Rate'] *100
        tp_rate_young = metrics_young[disease]['TP Rate'] *100
        fp_rate_old = metrics_old[disease]['FP Rate'] *100
        fp_rate_young = metrics_young[disease]['FP Rate'] *100
        fn_rate_old = metrics_old[disease]['FN Rate'] *100
        fn_rate_young = metrics_young[disease]['FN Rate'] *100


        # kl1 = metrics_old[disease]['KL div'] *100
        # kl2 = metrics_young[disease]['KL div'] *100

        
        # delta_KL_age = abs(kl1 - kl2)

        
        # Calculate delta AUC and equality of odds
        delta_auc_age = abs(auc_old - auc_young)
        eq_odds_age = 0.5 * (abs(tp_rate_old - tp_rate_young) + abs(fp_rate_old - fp_rate_young))
        
        fnr_diff_age = abs(fn_rate_old - fn_rate_young)
        
        # Append to the data list
        data_age.append([disease, auprc_overall, auc_overall, auc_old, auc_young, delta_auc_age, eq_odds_age, fnr_diff_age])

    # Create a DataFrame
    df_age = pd.DataFrame(data_age, columns=['Disease', 'AUPRC', 'AUC', 'AUC_old', 'AUC_young', 'Delta AUC', 'EqOdds', 'FNR_diff'])


    # Initialize an empty list to store the data
    data = []
    i = 0

    # Assuming 'metrics', 'df_sex', 'df_race', 'df_age', and 'df_health' are predefined and correctly structured
    # Iterate over the diseases in the metrics dictionary
    for disease, values in metrics.items():
        auprc_overall = values['AUPRC'] * 100
        auc_overall = values['AUC'] * 100

        # Append to the data list
        data.append([disease, auprc_overall, auc_overall] +
                    [df_sex['Delta AUC'][i], df_sex['EqOdds'][i], df_sex['FNR_diff'][i]] +
                    [df_race['Max Delta AUC'][i], df_race['Max EqOdds'][i], df_race['FNR_diff'][i]] +
                    [df_age['Delta AUC'][i], df_age['EqOdds'][i], df_age['FNR_diff'][i]])
                    # [df_health['Max Delta AUC'][i], df_health['Max EqOdds'][i], df_health['KL div'][i]])
        i += 1

    # Create a DataFrame
    columns = ['Disease', 'AUPRC', 'AUC', 'Delta AUPRC sex', 'EqOdds sex', 'Delta FNR sex',
            'Delta AUPRC race', 'EqOdds race', 'Delta FNR race', 'Delta AUPRC age', 'EqOdds age', 'Delta FNR age']
    df = pd.DataFrame(data, columns=columns)

    # Styling the DataFrame
    styled_df = df.style.format({
        'AUPRC': "{:.1f}",
        'AUC': "{:.1f}",
        'Delta AUPRC sex': "{:.1f}",
        'EqOdds sex': "{:.1f}",
        'Delta FNR sex': "{:.1f}",
        'Delta AUPRC race': "{:.1f}",
        'EqOdds race': "{:.1f}",
        'Delta FNR race': "{:.1f}",
        'Delta AUPRC age': "{:.1f}",
        'EqOdds age': "{:.1f}",
        'Delta FNR age': "{:.1f}"
        # 'Delta AUPRC health': "{:.1f}",
        # 'EqOdds health': "{:.1f}",
        # 'KL div health': "{:.1f}"
    }).background_gradient(cmap='OrRd', subset=[
        'AUPRC', 'AUC', 'Delta AUPRC sex', 'EqOdds sex', 'Delta FNR sex', 'Delta AUPRC race', 'EqOdds race', 'Delta FNR race',
        'Delta AUPRC age', 'EqOdds age', 'Delta FNR age'
    ])

    return styled_df, df




def bias_table_auprc(metrics, metrics_female, metrics_male, metrics_white, metrics_black, metrics_asian, metrics_young, metrics_old, race = False):

    # Initialize an empty list to store the data
    data_sex = []

    # Iterate over the diseases in the metrics dictionary
    for disease, values in metrics.items():
        # Extract AUC and rates from dictionaries
        
        auprc_overall = values['AUPRC'] *100

        auc_overall = values['AUC'] *100
        auprc_male = metrics_male[disease]['AUPRC'] *100
        auprc_female = metrics_female[disease]['AUPRC'] *100
        tp_rate_male = metrics_male[disease]['TP Rate'] *100
        tp_rate_female = metrics_female[disease]['TP Rate'] *100
        fp_rate_male = metrics_male[disease]['FP Rate'] *100
        fp_rate_female = metrics_female[disease]['FP Rate'] *100
        fn_rate_male = metrics_male[disease]['FN Rate'] *100
        fn_rate_female = metrics_female[disease]['FN Rate'] *100

        # kl1 = metrics_male[disease]['KL div'] *100
        # kl2 = metrics_female[disease]['KL div'] *100

        
        # delta_KL_sex = abs(kl1 - kl2)


        # Calculate delta AUC and equality of odds
        delta_auc_sex = abs(auprc_male - auprc_female) 
        eq_odds_sex = 0.5 * (abs(tp_rate_male - tp_rate_female) + abs(fp_rate_male - fp_rate_female))
        fnr_diff = abs(fn_rate_male - fn_rate_female)
        
        # Append to the data list
        data_sex.append([disease, auprc_overall, auc_overall, auprc_male, auprc_female, delta_auc_sex, eq_odds_sex, fnr_diff])

    # Create a DataFrame
    df_sex = pd.DataFrame(data_sex, columns=['Disease', 'AUPRC', 'AUC', 'AUC_Male', 'AUC_Female', 'Delta AUC', 'EqOdds', 'FNR_diff'])

    if race:
        data_race = []

        # Iterate over the diseases in the metrics dictionary
        for disease, values in metrics.items():
            # Extract AUC and rates from dictionaries
            
            auprc_overall = values['AUPRC'] *100

            auc_overall = values['AUC'] *100
            auprc_male = metrics_white[disease]['AUPRC'] *100
            auprc_female = metrics_asian[disease]['AUPRC'] *100
            tp_rate_male = metrics_white[disease]['TP Rate'] *100
            tp_rate_female = metrics_asian[disease]['TP Rate'] *100
            fp_rate_male = metrics_white[disease]['FP Rate'] *100
            fp_rate_female = metrics_asian[disease]['FP Rate'] *100
            fn_rate_male = metrics_white[disease]['FN Rate'] *100
            fn_rate_female = metrics_asian[disease]['FN Rate'] *100

            # kl1 = metrics_white[disease]['KL div'] *100
            # kl2 = metrics_asian[disease]['KL div'] *100

            
            # delta_KL_sex = abs(kl1 - kl2)


            # Calculate delta AUC and equality of odds
            delta_auc_sex = abs(auprc_male - auprc_female) 
            eq_odds_sex = 0.5 * (abs(tp_rate_male - tp_rate_female) + abs(fp_rate_male - fp_rate_female))
            fnr_diff = abs(fn_rate_male - fn_rate_female)
            
            # Append to the data list
            data_race.append([disease, auprc_overall, auc_overall, auprc_male, auprc_female, delta_auc_sex, eq_odds_sex, fnr_diff])

        # Create a DataFrame
        df_race = pd.DataFrame(data_race, columns=['Disease', 'AUPRC', 'AUC', 'AUC_White', 'AUC_Asian', 'Delta AUC', 'EqOdds', 'FNR_diff'])


    else:
        # Initialize an empty list to store the data
        data_race = []

        # Iterate over the diseases in the metrics dictionary
        for disease, values in metrics.items():
            auprc_overall = values['AUPRC'] *100
            auc_overall = values['AUC'] *100
            auprc_groups = [
                metrics_white[disease]['AUPRC'] *100,
                metrics_black[disease]['AUPRC'] *100,
                metrics_asian[disease]['AUPRC'] *100
            ]
            tp_rates = [
                metrics_white[disease]['TP Rate'] *100,
                metrics_black[disease]['TP Rate'] *100,
                metrics_asian[disease]['TP Rate'] *100
            ]
            fp_rates = [
                metrics_white[disease]['FP Rate'] *100,
                metrics_black[disease]['FP Rate'] *100,
                metrics_asian[disease]['FP Rate'] *100
            ]

            fn_rates = [
                metrics_white[disease]['FN Rate'] *100,
                metrics_black[disease]['FN Rate'] *100,
                metrics_asian[disease]['FN Rate'] *100
            ]

            # kl_rates = [
            #     metrics_white[disease]['KL div'] *100,
            #     metrics_black[disease]['KL div'] *100,
            #     metrics_asian[disease]['KL div'] *100
            # ]

            # delta_kl_race = max(abs(kl_rates[i] - kl_rates[j]) for i in range(len(kl_rates)) for j in range(i + 1, len(kl_rates)))

            # Calculate the maximum delta AUC
            delta_auprc_race = max(abs(auprc_groups[i] - auprc_groups[j]) for i in range(len(auprc_groups)) for j in range(i + 1, len(auprc_groups)))

            # Calculate the maximum equality of odds
            eq_odds_race = max(
                0.5 * (abs(tp_rates[i] - tp_rates[j]) + abs(fp_rates[i] - fp_rates[j]))
                for i in range(len(tp_rates)) for j in range(i + 1, len(tp_rates))
            )

            fnr_diff_race = max(
                abs(fn_rates[i] - fn_rates[j])
                for i in range(len(fn_rates)) for j in range(i + 1, len(fn_rates))
            )

            # Append to the data list
            data_race.append([disease, auprc_overall, auc_overall] + auprc_groups + [delta_auprc_race, eq_odds_race, fnr_diff_race])

        # Create a DataFrame
        columns = ['Disease', 'AUPRC', 'AUC', 'AUC_White', 'AUC_Black', 'AUC_Asian', 'Max Delta AUC', 'Max EqOdds', 'FNR_diff']
        df_race = pd.DataFrame(data_race, columns=columns)


    # Initialize an empty list to store the data
    data_age = []

    # Iterate over the diseases in the metrics dictionary
    for disease, values in metrics.items():
        # Extract AUC and rates from dictionaries
        
        auprc_overall = values['AUPRC'] *100

        auc_overall = values['AUC'] *100
        auc_old = metrics_old[disease]['AUPRC'] *100
        auc_young = metrics_young[disease]['AUPRC'] *100
        tp_rate_old = metrics_old[disease]['TP Rate'] *100
        tp_rate_young = metrics_young[disease]['TP Rate'] *100
        fp_rate_old = metrics_old[disease]['FP Rate'] *100
        fp_rate_young = metrics_young[disease]['FP Rate'] *100
        fn_rate_old = metrics_old[disease]['FN Rate'] *100
        fn_rate_young = metrics_young[disease]['FN Rate'] *100


        # kl1 = metrics_old[disease]['KL div'] *100
        # kl2 = metrics_young[disease]['KL div'] *100

        
        # delta_KL_age = abs(kl1 - kl2)

        
        # Calculate delta AUC and equality of odds
        delta_auc_age = abs(auc_old - auc_young)
        eq_odds_age = 0.5 * (abs(tp_rate_old - tp_rate_young) + abs(fp_rate_old - fp_rate_young))
        
        fnr_diff_age = abs(fn_rate_old - fn_rate_young)
        
        # Append to the data list
        data_age.append([disease, auprc_overall, auc_overall, auc_old, auc_young, delta_auc_age, eq_odds_age, fnr_diff_age])

    # Create a DataFrame
    df_age = pd.DataFrame(data_age, columns=['Disease', 'AUPRC', 'AUC', 'AUC_old', 'AUC_young', 'Delta AUC', 'EqOdds', 'FNR_diff'])


    # Initialize an empty list to store the data
    data = []
    i = 0

    # Assuming 'metrics', 'df_sex', 'df_race', 'df_age', and 'df_health' are predefined and correctly structured
    # Iterate over the diseases in the metrics dictionary
    for disease, values in metrics.items():
        auprc_overall = values['AUPRC'] * 100
        auc_overall = values['AUC'] * 100

        # Append to the data list
        data.append([disease, auprc_overall, auc_overall] +
                    [df_sex['Delta AUC'][i]] + [df_race['Max Delta AUC'][i]] + [df_age['Delta AUC'][i]])
                    # [df_health['Max Delta AUC'][i], df_health['Max EqOdds'][i], df_health['KL div'][i]])
        i += 1

    # Create a DataFrame
    columns = ['Disease', 'AUPRC', 'AUC', 'Delta AUPRC sex', 'Delta AUPRC race', 'Delta AUPRC age']
    df = pd.DataFrame(data, columns=columns)

    # Styling the DataFrame
    styled_df = df.style.format({
        'AUPRC': "{:.1f}",
        'AUC': "{:.1f}",
        'Delta AUPRC sex': "{:.1f}",
        'Delta AUPRC race': "{:.1f}",
        'Delta AUPRC age': "{:.1f}",
        # 'Delta AUPRC health': "{:.1f}",
        # 'EqOdds health': "{:.1f}",
        # 'KL div health': "{:.1f}"
    }).background_gradient(cmap='OrRd', subset=[
        'AUPRC', 'AUC', 'Delta AUPRC sex', 'Delta AUPRC race', 'Delta AUPRC age'
    ])

    return styled_df, df
