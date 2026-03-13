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
            print(f"\n[SKIP] Foundation Model {fm.get('name', 'Sconosciuto')} saltato per configurazione.\n")
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
                print(f"   [SKIP] Dataset {dataset['name']} saltato per configurazione.\n")
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

            for target_col in target_columns:
                for bias_attr in bias_attributes:

                    print(f"\n{'='*70}")
                    print(f"🚀 AVVIO ESPERIMENTO")
                    print(f"   Dataset: {ds_name}")
                    print(f"   Target:  {target_col}")
                    print(f"   Bias:    {bias_attr}")
                    print(f"{'='*70}\n")

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

                    print("=====================================================\n")
            
        if config.experiment.task5:
            ds = [d for d in datasets if d.get("run", True)]
        
            if len(ds) > 1:

                loader = DataLoader(config)
                combined_data = loader.load_multiple_datasets(ds)

                exp2 = Experiment(config)
                task5_results = exp2.run_task_etnia(combined_data, fm["name"])

                if config.experiment.save_results_csv:
                    MetricManager.save_domain_shift_to_csv(
                        fm_name=fm["name"],
                        model_type=config.experiment.model_type,
                        task5_results=task5_results
                    )

            else:
                print(f"\n[SKIP] Task 5 saltato perché non ci sono più dataset da combinare per il foundation model {fm['name']}.\n")
        else:
            print(f"\n[SKIP] Task 5 saltato per configurazione.\n")
