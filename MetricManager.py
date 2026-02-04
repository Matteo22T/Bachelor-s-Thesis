from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, accuracy_score
import numpy as np  

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