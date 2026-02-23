from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, accuracy_score
import numpy as np  
import pandas as pd
import os
from datetime import datetime

# Classe per la gestione delle metriche di performance e fairness
# Calcola metriche standard (Accuracy, F1-Score, AUROC) e metriche di equità (Max-Min Gap, Equality of Odds)
# Promemoria per ricordare cosa sono queste metriche:
# - Max-Min Gap: Differenza assoluta tra le performance (es. Accuracy) dei due gruppi. Un valore più basso indica maggiore equità.
# - Equality of Odds: Si riferisce alla parità nei tassi di veri positivi (TPR) e falsi positivi (FPR) tra i gruppi.
#   Un rapporto vicino a 1 indica maggiore equità
# - TPR Ratio: TPR(gruppo A) / TPR(gruppo B)
# - FPR Ratio: FPR(gruppo A) / FPR(gruppo B)
#  Un rapporto > 1 indica che il gruppo A ha un tasso più alto rispetto al gruppo B, e viceversa.
# - F1-Score: Media armonica di precisione e richiamo, utile per dataset sbilanciati.
# - AUROC: Area sotto la curva ROC, misura la capacità del modello di distinguere tra classi positive e negative.



# no media ma mediana e quartili
class MetricManager:

    @staticmethod
    def calculate_tpr_fpr(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  
        return tpr, fpr
    
    
    def compute_metrics(self, group_a_data, group_b_data):

        # 1. Metriche Standard per Gruppo A
        acc_a = accuracy_score(group_a_data['true'], group_a_data['pred'])
        f1_a = f1_score(group_a_data['true'], group_a_data['pred'])
        auc_a = roc_auc_score(group_a_data['true'], group_a_data['proba'])
        tpr_a, fpr_a = self.calculate_tpr_fpr(group_a_data['true'], group_a_data['pred'])

        # 2. Metriche Standard per Gruppo B
        acc_b = accuracy_score(group_b_data['true'], group_b_data['pred'])
        f1_b = f1_score(group_b_data['true'], group_b_data['pred'])
        auc_b = roc_auc_score(group_b_data['true'], group_b_data['proba'])
        tpr_b, fpr_b = self.calculate_tpr_fpr(group_b_data['true'], group_b_data['pred'])

        # Rapporto TPR: TPR(maschi)/TPR(femmine)
        tpr_ratio = tpr_a / tpr_b if tpr_b > 0 else 0
        # Rapporto FPR: FPR(maschi)/FPR(femmine)
        fpr_ratio = fpr_a / fpr_b if fpr_b > 0 else 0

        # --- MAX-MIN GAP ACCURACY ---
        acc_gap = abs(acc_a - acc_b)

        return {
            'performance': {
                'group_a': {'acc': acc_a, 'f1': f1_a, 'auroc': auc_a},
                'group_b': {'acc': acc_b, 'f1': f1_b, 'auroc': auc_b}
            },
            'fairness': {
                'eq_odds_tpr_ratio': tpr_ratio,
                'eq_odds_fpr_ratio': fpr_ratio,
                'max_min_gap_accuracy': acc_gap
            }
        }
    

    @staticmethod
    def get_median_iqr(data):
        """Calcola Mediana, 25° percentile (Q1) e 75° percentile (Q3)"""
        return np.median(data), np.percentile(data, 25), np.percentile(data, 75)

    @staticmethod
    def print_summary(task_name, results_list):
        """
        Stampa i risultati completi e le metriche di bias aggregati con Mediana e IQR.
        """
        if not results_list:
            print(f"Nessun risultato per {task_name}")
            return

        print(f"\n{'='*15} RISULTATI (Mediana e IQR): {task_name} {'='*15}")
        
        # Estrazione Metriche di Fairness
        gaps = [res['fairness']['max_min_gap_accuracy'] for res in results_list]
        tpr_ratios = [res['fairness']['eq_odds_tpr_ratio'] for res in results_list]
        fpr_ratios = [res['fairness']['eq_odds_fpr_ratio'] for res in results_list] 
        
        # Estrazione Metriche Maschi (Group A)
        acc_m = [res['performance']['group_a']['acc'] for res in results_list]
        f1_m = [res['performance']['group_a']['f1'] for res in results_list]       
        auc_m = [res['performance']['group_a']['auroc'] for res in results_list]    

        # Estrazione Metriche Femmine (Group B)
        acc_f = [res['performance']['group_b']['acc'] for res in results_list]
        f1_f = [res['performance']['group_b']['f1'] for res in results_list]
        auc_f = [res['performance']['group_b']['auroc'] for res in results_list]

        # Calcolo Statistiche tramite il metodo statico della classe
        med_acc_m, q1_acc_m, q3_acc_m = MetricManager.get_median_iqr(acc_m)
        med_f1_m, q1_f1_m, q3_f1_m = MetricManager.get_median_iqr(f1_m)
        med_auc_m, q1_auc_m, q3_auc_m = MetricManager.get_median_iqr(auc_m)

        med_acc_f, q1_acc_f, q3_acc_f = MetricManager.get_median_iqr(acc_f)
        med_f1_f, q1_f1_f, q3_f1_f = MetricManager.get_median_iqr(f1_f)
        med_auc_f, q1_auc_f, q3_auc_f = MetricManager.get_median_iqr(auc_f)

        med_gap, q1_gap, q3_gap = MetricManager.get_median_iqr(gaps)
        med_tpr, q1_tpr, q3_tpr = MetricManager.get_median_iqr(tpr_ratios)
        med_fpr, q1_fpr, q3_fpr = MetricManager.get_median_iqr(fpr_ratios)

        # Stampa a schermo formattata
        print("---------------------------------  PERFORMANCE (Mediana [IQR])  ---------------------------------")
        print(f"{'Metrica':<15} | {'Maschi (Group A)':<30} | {'Femmine (Group B)':<30} |")
        print("-" * 83)
        print(f"{'Accuracy':<15} | {med_acc_m*100:.2f}% [{q1_acc_m*100:.2f} - {q3_acc_m*100:.2f}] | {med_acc_f*100:.2f}% [{q1_acc_f*100:.2f} - {q3_acc_f*100:.2f}] |")
        print(f"{'F1-Score':<15} | {med_f1_m:.4f} [{q1_f1_m:.4f} - {q3_f1_m:.4f}]   | {med_f1_f:.4f} [{q1_f1_f:.4f} - {q3_f1_f:.4f}]   |")
        print(f"{'AUROC':<15} | {med_auc_m:.4f} [{q1_auc_m:.4f} - {q3_auc_m:.4f}]   | {med_auc_f:.4f} [{q1_auc_f:.4f} - {q3_auc_f:.4f}]   |")
        
        print(f"\n--- FAIRNESS & BIAS (Indicatori di equità - Mediana [IQR]) ---")
        print(f"-> Max-Min Gap Accuracy:     {med_gap*100:.2f}% [{q1_gap*100:.2f} - {q3_gap*100:.2f}]  (Ideale: 0%)")
        print(f"-> TPR Ratio (Eq. Odds):     {med_tpr:.4f} [{q1_tpr:.4f} - {q3_tpr:.4f}]      (Ideale: 1.0)")
        print(f"-> FPR Ratio (Eq. Odds):     {med_fpr:.4f} [{q1_fpr:.4f} - {q3_fpr:.4f}]      (Ideale: 1.0)")
        print("=================================================================================\n")


    
    @staticmethod
    def save_to_csv(task_name, model_type, results_list, csv_path):
        """Salva i risultati aggregati (Mediana) in un file CSV in modalità append."""
        if not results_list:
            return

        # Estrazione Metriche (identica a print_summary)
        gaps = [res['fairness']['max_min_gap_accuracy'] for res in results_list]
        tpr_ratios = [res['fairness']['eq_odds_tpr_ratio'] for res in results_list]
        fpr_ratios = [res['fairness']['eq_odds_fpr_ratio'] for res in results_list] 
        
        acc_m = [res['performance']['group_a']['acc'] for res in results_list]
        f1_m = [res['performance']['group_a']['f1'] for res in results_list]       
        auc_m = [res['performance']['group_a']['auroc'] for res in results_list]    

        acc_f = [res['performance']['group_b']['acc'] for res in results_list]
        f1_f = [res['performance']['group_b']['f1'] for res in results_list]
        auc_f = [res['performance']['group_b']['auroc'] for res in results_list]

        # Calcolo Statistiche (Mediana)
        med_acc_m, q1_acc_m, q3_acc_m = MetricManager.get_median_iqr(acc_m)
        med_f1_m, q1_f1_m, q3_f1_m = MetricManager.get_median_iqr(f1_m)
        med_auc_m, q1_auc_m, q3_auc_m = MetricManager.get_median_iqr(auc_m)

        med_acc_f, q1_acc_f, q3_acc_f = MetricManager.get_median_iqr(acc_f)
        med_f1_f, q1_f1_f, q3_f1_f = MetricManager.get_median_iqr(f1_f)
        med_auc_f, q1_auc_f, q3_auc_f = MetricManager.get_median_iqr(auc_f)

        med_gap, q1_gap, q3_gap = MetricManager.get_median_iqr(gaps)
        med_tpr, q1_tpr, q3_tpr = MetricManager.get_median_iqr(tpr_ratios)
        med_fpr, q1_fpr, q3_fpr = MetricManager.get_median_iqr(fpr_ratios)

        # Creazione della riga di log
        row = {
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Model': model_type,
            'Task': task_name,
            
            # Bias Metrics
            'Gap_Accuracy_Med': round(med_gap * 100, 2),
            'Gap_Accuracy_Q1': round(q1_gap * 100, 2),
            'Gap_Accuracy_Q3': round(q3_gap * 100, 2),
            
            'TPR_Ratio_Med': round(med_tpr, 4),
            'TPR_Ratio_Q1': round(q1_tpr, 4),
            'TPR_Ratio_Q3': round(q3_tpr, 4),
            
            'FPR_Ratio_Med': round(med_fpr, 4),
            'FPR_Ratio_Q1': round(q1_fpr, 4),
            'FPR_Ratio_Q3': round(q3_fpr, 4),
            
            # Male (Group A) Metrics
            'Acc_Male_Med': round(med_acc_m * 100, 2),
            'Acc_Male_Q1': round(q1_acc_m * 100, 2),
            'Acc_Male_Q3': round(q3_acc_m * 100, 2),
            
            'F1_Male_Med': round(med_f1_m, 4),
            'F1_Male_Q1': round(q1_f1_m, 4),
            'F1_Male_Q3': round(q3_f1_m, 4),

            'AUC_Male_Med': round(med_auc_m, 4),
            'AUC_Male_Q1': round(q1_auc_m, 4),
            'AUC_Male_Q3': round(q3_auc_m, 4),
            
            # Female (Group B) Metrics
            'Acc_Female_Med': round(med_acc_f * 100, 2),
            'Acc_Female_Q1': round(q1_acc_f * 100, 2),
            'Acc_Female_Q3': round(q3_acc_f * 100, 2),
            
            'F1_Female_Med': round(med_f1_f, 4),
            'F1_Female_Q1': round(q1_f1_f, 4),
            'F1_Female_Q3': round(q3_f1_f, 4),

            'AUC_Female_Med': round(med_auc_f, 4),
            'AUC_Female_Q1': round(q1_auc_f, 4),
            'AUC_Female_Q3': round(q3_auc_f, 4)
        }

        df = pd.DataFrame([row])
        
        # Creazione cartella se non esiste
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # Salvataggio: se il file esiste già aggiunge la riga (append), altrimenti lo crea con l'header
        file_exists = os.path.exists(csv_path)
        df.to_csv(csv_path, mode='a', index=False, header=not file_exists)