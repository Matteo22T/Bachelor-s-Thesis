from ConfigLoader import ConfigLoader
from DataLoader import DataLoader
from Experiment import Experiment
from MetricManager import MetricManager
import numpy as np
import copy

import warnings
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
# Ignora gli avvisi di mancata convergenza
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# Ignora gli avvisi delle combinazioni incompatibili nella GridSearch (lbfgs + l1)
warnings.filterwarnings("ignore", category=FitFailedWarning)
# Ignora gli UserWarning generali (quelli fastidiosi sull' l1_ratio deprecato)
warnings.filterwarnings("ignore", category=UserWarning)



if __name__ == "__main__":
    
    config = ConfigLoader("config.yaml")

    foundation_models = config.data.foundation_models
    target_columns = config.experiment.target_column
    bias_attributes = config.experiment.bias_attribute

    for fm in foundation_models:
        if not fm.get("run", True):
            continue

        print(f"\n{'='*80}")
        print(f"FOUNDATION MODEL: {fm['name']}")
        print(f"{'='*80}\n")

        datasets = fm.get("datasets", [])


        for dataset in datasets:
            print(f"\n{'='*80}")
            print(f"DATASET: {dataset['name']}")
            print(f"{'='*80}\n")

            if not dataset.get("run", True):
                continue

            ds_name = dataset['name']
            embedding_path = dataset['embedding_path']
            metadata_path = dataset['metadata_path']


            temp_config = copy.deepcopy(config)
            temp_config.data.current_dataset = ds_name
            temp_config.data.embeddings_path = embedding_path
            temp_config.data.metadata_path = metadata_path
            
            loader_iniziale = DataLoader(temp_config)
            data = loader_iniziale.load_data()

            data = loader_iniziale.filter_correlated_features(data)

            for target_col in target_columns:
                for bias_attr in bias_attributes:

                    current_config = copy.deepcopy(config)

                    current_config.data.current_dataset = ds_name
                    current_config.data.embeddings_path = embedding_path
                    current_config.data.metadata_path = metadata_path

                    current_config.experiment.target_column = [target_col]
                    current_config.experiment.bias_attribute = [bias_attr]

                    save_csv = current_config.experiment.save_results_csv
                    csv_path = current_config.experiment.results_csv_path
                    model_type = current_config.experiment.model_type

                    full_data = data.copy()

                    exp = Experiment(current_config)
        

                    if exp.config.experiment.task1:
                        results_base = exp.run_base_case(full_data)
                        MetricManager.print_summary("CASO BASE (Task 1)", results_base)
                        if save_csv:
                            MetricManager.save_to_csv(ds_name, target_col, bias_attr, "CASO BASE (Task 1)", model_type, results_base, csv_path)

                    if exp.config.experiment.task2:
                        results_extreme = exp.run_extreme_case(full_data)
                        MetricManager.print_summary("CASO ESTREMO (Task 2)", results_extreme)
                        if save_csv:
                            MetricManager.save_to_csv(ds_name, target_col, bias_attr, "CASO ESTREMO (Task 2)", model_type, results_extreme, csv_path)

                    if exp.config.experiment.task3:
                        exp.run_counterfactual_case(full_data)

                    if exp.config.experiment.task4:
                        results_shap_dict = exp.run_shap_analysis(full_data)
                        
                        for n_feat, results_shap in results_shap_dict.items():
                            
                            task_name = f"TASK 4 (Ablation: {n_feat} proxy rimosse)"
                            
                            MetricManager.print_summary(task_name, results_shap)
                            
                            if save_csv:
                                MetricManager.save_to_csv(ds_name, target_col, bias_attr, task_name, model_type, results_shap, csv_path)

                    if config.experiment.task6:
                        exp_corr = Experiment(config)
                        exp_corr.run_correlation_analysis(full_data, fm["name"], ds_name)


                    if config.experiment.task7:
                        results_eff_dict = exp.run_shap_efficiency_analysis(full_data)
                        
                        for n_rem, results_eff in results_eff_dict.items():
                            task_label = f"TASK 7 (Efficienza: -{n_rem} feature)"
                            MetricManager.print_summary(task_label, results_eff)
                            if save_csv:
                                MetricManager.save_to_csv(ds_name, target_col, bias_attr, task_label, model_type, results_eff, csv_path)


                    if config.experiment.task8:
                        results_task8 = exp.run_overlap_analysis(full_data)
                        
                        if results_task8:
                            task_label = "TASK 8 (Mitigazione Conflitti)"
                            MetricManager.print_summary(task_label, results_task8)
                            
                            if save_csv:
                                MetricManager.save_to_csv(ds_name, target_col, bias_attr, task_label, 
                                                        model_type, results_task8, csv_path)  

                    if config.experiment.task9:
                        audit_results = exp.run_bias_audit(full_data)

                        if config.experiment.save_results_csv:
                            MetricManager.save_bias_audit_to_csv(
                                ds_name=ds_name,
                                model_type=model_type,
                                audit_results=audit_results
                            )

                    print("=====================================================\n")


            
        if config.experiment.task5:
            ds = [d for d in datasets if d.get("run", True)]
        
            if len(ds) > 1:

                loader = DataLoader(config)
                combined_data = loader.load_multiple_datasets(ds)

                combined_data = loader.filter_correlated_features(combined_data)

                exp2 = Experiment(config)
                task5_results = exp2.run_task_etnia(combined_data, fm["name"])

                if config.experiment.save_results_csv:
                    MetricManager.save_domain_shift_to_csv(
                        fm_name=fm["name"],
                        model_type=config.experiment.model_type,
                        task5_results=task5_results
                    )