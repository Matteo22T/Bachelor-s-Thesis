from Experiment import Experiment
from MetricManager import MetricManager
import numpy as np

import warnings
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
# Ignora gli avvisi di mancata convergenza
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# Ignora gli avvisi delle combinazioni incompatibili nella GridSearch (lbfgs + l1)
warnings.filterwarnings("ignore", category=FitFailedWarning)
# Ignora gli UserWarning generali (quelli fastidiosi sull' l1_ratio deprecato)
warnings.filterwarnings("ignore", category=UserWarning)



if __name__ == "__main__":
    exp = Experiment(config_path='config.yaml')
    
    full_data = exp.data_loader.load_data()

    save_csv = getattr(exp.config.experiment, 'save_results_csv', False)
    csv_path = getattr(exp.config.experiment, 'results_csv_path', 'results.csv')
    model_type = exp.config.experiment.model_type

    if exp.config.experiment.task1:
        results_base = exp.run_base_case(full_data)
        MetricManager.print_summary("CASO BASE (Task 1)", results_base)
        if save_csv:
            MetricManager.save_to_csv("CASO BASE (Task 1)", model_type, results_base, csv_path)

    if exp.config.experiment.task2:
        results_extreme = exp.run_extreme_case(full_data)
        MetricManager.print_summary("CASO ESTREMO (Task 2)", results_extreme)
        if save_csv:
            MetricManager.save_to_csv("CASO ESTREMO (Task 2)", model_type, results_extreme, csv_path)

    if exp.config.experiment.task3:
        exp.run_counterfactual_case(full_data)

    if exp.config.experiment.task4:
        results_shap_dict = exp.run_shap_analysis(full_data)
        
        for n_feat, results_shap in results_shap_dict.items():
            
            task_name = f"TASK 4 (Ablation: {n_feat} proxy rimosse)"
            
            MetricManager.print_summary(task_name, results_shap)
            
            if save_csv:
                MetricManager.save_to_csv(task_name, model_type, results_shap, csv_path)

    print("=====================================================\n")

