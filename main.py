from Experiment import Experiment
import numpy as np

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def print_average_results(task_name, results_list):
    """
    Stampa i risultati completi (Acc, F1, AUROC) e le metriche di bias (Gap, Ratio).
    """
    if not results_list:
        print(f"Nessun risultato per {task_name}")
        return

    print(f"\n{'='*10} RISULTATI MEDI: {task_name} {'='*10}")
    
    # Metriche di Fairness
    gaps = [res['fairness']['max_min_gap_accuracy'] for res in results_list]
    tpr_ratios = [res['fairness']['eq_odds_tpr_ratio'] for res in results_list]
    fpr_ratios = [res['fairness']['eq_odds_fpr_ratio'] for res in results_list] # 
    
    # Metriche Maschi (Group A)
    acc_m = [res['performance']['group_a']['acc'] for res in results_list]
    f1_m = [res['performance']['group_a']['f1'] for res in results_list]       # 
    auc_m = [res['performance']['group_a']['auroc'] for res in results_list]    # 

    # Metriche Femmine (Group B)
    acc_f = [res['performance']['group_b']['acc'] for res in results_list]
    f1_f = [res['performance']['group_b']['f1'] for res in results_list]
    auc_f = [res['performance']['group_b']['auroc'] for res in results_list]

    print("-------------------  PERFORMANCE  -------------------")
    print(f"{'Metrica':<15} | {'Maschi':<10} | {'Femmine':<10} |")
    print("-" * 43)
    print(f"{'Accuracy':<15} | {np.mean(acc_m)*100:.2f}%     | {np.mean(acc_f)*100:.2f}%     |")
    print(f"{'F1-Score':<15} | {np.mean(f1_m):.4f}     | {np.mean(f1_f):.4f}     |")
    print(f"{'AUROC':<15}    | {np.mean(auc_m):.4f}     | {np.mean(auc_f):.4f}     |")
    
    print(f"\n--- FAIRNESS & BIAS (Indicatori di equità) ---")
    
    # Max-Min Gap 
    gap_medio = np.mean(gaps)
    print(f"-> Max-Min Gap Accuracy:     {gap_medio*100:.2f}%  (Ideale: 0%)")
    
    # Equality of Odds (TPR) 
    # Se > 1: Favorisce Maschi. Se < 1: Favorisce Femmine.
    tpr_mean = np.mean(tpr_ratios)
    print(f"-> TPR Ratio (Eq. Odds):     {tpr_mean:.4f}   (Ideale: 1.0)")
    
    # Equality of Odds (FPR) 
    # Se > 1: Favorisce Femmine. Se < 1: Favorisce Maschi.
    fpr_mean = np.mean(fpr_ratios)
    print(f"-> FPR Ratio (Eq. Odds):     {fpr_mean:.4f}   (Ideale: 1.0)")
    print("=====================================================\n")



if __name__ == "__main__":
    exp = Experiment(config_path='config.yaml')
    
    full_data = exp.data_loader.load_data()

    results_base = exp.run_base_case(full_data)
    print_average_results("CASO BASE (Task 1)", results_base)    

    results_extreme = exp.run_extreme_case(full_data)
    print_average_results("CASO ESTREMO (Task 2)", results_extreme)

    results_shap = exp.run_shap_analysis(full_data)
    print_average_results("TASK 4 (SHAP Ablation)", results_shap)

    exp.run_counterfactual_case(full_data)
    print("=====================================================\n")

