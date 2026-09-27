from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from DataLoader import DataLoader
from MetricManager import MetricManager
from ShapExplainer import ShapFactory
from Classifier import LogisticRegressionClassifier, RandomForestClassifierModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

class Experiment:
    def __init__(self, config):
        self.config = config
        self.data_loader = DataLoader(self.config)
        self.metric_manager = MetricManager()
 
    def _get_classifier(self):
        model_type = self.config.experiment.model_type
        if model_type == "random_forest":
            return RandomForestClassifierModel(self.config)
        elif model_type == "logistic_regression":
            return LogisticRegressionClassifier(self.config)
        else:
            raise ValueError(f"Model type '{model_type}' not recognized.")
        
    def run_fold_logic(self, X_train, y_train, X_test, y_test, meta_test, save_path=None):

        classifier = self._get_classifier()
        classifier.train(X_train, y_train, model_save_path=save_path)

        # Divisione del test set per attributo di bias
        bias_col = self.config.experiment.bias_attribute[0]
        unique_vals = sorted(meta_test[bias_col].dropna().unique())

        val_a = unique_vals[0]
        val_b = unique_vals[1]
        
        mask_a = meta_test[bias_col] == val_a
        mask_b = meta_test[bias_col] == val_b

        X_test_a = X_test[mask_a]
        y_test_a = y_test[mask_a]

        X_test_b = X_test[mask_b]
        y_test_b = y_test[mask_b]

        preds_a = classifier.predict(X_test_a)
        preds_b = classifier.predict(X_test_b)

        # Probabilità per calcolo AUROC
        proba_a = classifier.predict_proba(X_test_a)[:, 1]
        proba_b = classifier.predict_proba(X_test_b)[:, 1]

        # Gruppi per calcolo metriche
        group_a = {'true': y_test_a, 'pred': preds_a, 'proba': proba_a}
        group_b = {'true': y_test_b, 'pred': preds_b, 'proba': proba_b}

        results = self.metric_manager.compute_metrics(group_a, group_b)
        return results
    

    # Task 1 Caso Base: Addestramento su dati bilanciati di entrambi i sessi e inferenza su entrambi i sessi separatamente.
    # Obiettivo: Valutare performance e bias in condizioni standard.
    def run_base_case(self,data):
       print("Avvio caso base...")
       balanced_data = self.data_loader.balance_model(data)

       X, y = self.data_loader.prepare_features(balanced_data, train=True)

       meta_data = balanced_data

       sss = StratifiedShuffleSplit(n_splits=self.config.experiment.n_splits,
                                    test_size=self.config.experiment.test_size,
                                    random_state=self.config.experiment.random_seed)
       
       
       all_results = []

       for fold, (train_index, test_index) in enumerate(sss.split(X, y)):
           
           # Divisione dati
           X_train, X_test = X[train_index], X[test_index]
           y_train, y_test = y[train_index], y[test_index]

           # Divisione metadati        
           meta_test = meta_data.iloc[test_index]

           ds_name = self.config.data.current_dataset
           target = self.config.experiment.target_column[0]
           bias = self.config.experiment.bias_attribute[0]

           base_dir = f"checkpoints/{self.config.experiment.model_type}/{ds_name}/{target}_{bias}/base_case"
           os.makedirs(base_dir, exist_ok=True)
           checkpoint_path = f"{base_dir}/fold_{fold}.joblib"

           metrics = self.run_fold_logic(X_train, y_train, X_test, y_test, meta_test, save_path=checkpoint_path)
           all_results.append(metrics)

           #print(f"Fold {fold + 1}/{self.config.experiment.n_splits} completato.") 

       #print(f"Esperimento completato su {len(all_results)} iterazioni.")

       return all_results
    
    #Task 2Caso Estremo: Addestramento solo su un sesso (es. maschi) e inferenza su entrambi i sessi separatamente.
    #Obiettivo: Vedere quanto il modello "soffre" se non vede mai una donna durante lo studio.
    def run_extreme_case(self, data):
        print("Avvio caso estremo...")
        balanced_data = self.data_loader.balance_model(data)

        X, y = self.data_loader.prepare_features(balanced_data, train=True)

        meta_data = balanced_data

        sss = StratifiedShuffleSplit(n_splits=self.config.experiment.n_splits,
                                     test_size=self.config.experiment.test_size,
                                     random_state=self.config.experiment.random_seed)
        
        all_results = []

        for fold, (train_index, test_index) in enumerate(sss.split(X, y)):
            
            X_train_full, X_test = X[train_index], X[test_index]
            y_train_full, y_test = y[train_index], y[test_index]

            meta_train = meta_data.iloc[train_index]
            meta_test = meta_data.iloc[test_index]

            train_mask = meta_train['sex'] == self.config.experiment.female_value

            X_train = X_train_full[train_mask]
            y_train = y_train_full[train_mask]

            ds_name = self.config.data.current_dataset
            target = self.config.experiment.target_column[0]
            bias = self.config.experiment.bias_attribute[0]

            base_dir = f"checkpoints/{self.config.experiment.model_type}/{ds_name}/{target}_{bias}/extreme_case_female"
            os.makedirs(base_dir, exist_ok=True)
            checkpoint_path = f"{base_dir}/fold_{fold}.joblib"

            metrics = self.run_fold_logic(X_train, y_train, X_test, y_test, meta_test, save_path=checkpoint_path)
            all_results.append(metrics)

            #print(f"Fold {fold + 1}/{self.config.experiment.n_splits} completato.")
        
        #print(f"Esperimento completato su {len(all_results)} iterazioni.")
        return all_results



    #Task 4 Analisi SHAP e Rimozione Feature Proxy: Utilizza SHAP per identificare le feature più correlate al sesso e rimuovile iterativamente, valutando l'impatto sulle performance e sul bias.
    def run_shap_analysis(self,data):
        print("--- Avvio Analisi SHAP e Rimozione Feature (Task 4) ---")

        X, _ = self.data_loader.prepare_features(data, train=True)
        y_sex = data["sex"].values
        model_type = self.config.experiment.model_type

        ds_name = self.config.data.current_dataset
        target = self.config.experiment.target_column[0]
        bias = self.config.experiment.bias_attribute[0]
        base_dir = f"checkpoints/{model_type}/{ds_name}/{target}_{bias}/shap_ablation"
        os.makedirs(base_dir, exist_ok=True)

        classifier = self._get_classifier()
        shap_model_path = f"{base_dir}/shap_detector_model.joblib"
        classifier.train(X, y_sex, model_save_path=shap_model_path)

        shap_cache_path = f"{base_dir}/feature_importance.csv"
        force_retrain = getattr(self.config.experiment, 'force_retrain', False)

        if os.path.exists(shap_cache_path) and not force_retrain:
            # Se la classifica esiste già, la carichiamo istantaneamente dal CSV
            feature_importance = pd.read_csv(shap_cache_path)

        else:
           
            print("Calcolo dei valori SHAP in corso... (Potrebbe richiedere un po' di tempo)")
            sample_size = self.config.experiment.sample_size_shap
            
            if sample_size < len(X):
                _, X_sample, _, _ = train_test_split(
                    X, y_sex, 
                    test_size=sample_size, 
                    stratify=y_sex, 
                    random_state=self.config.experiment.random_seed
                )
            else:
                X_sample = X

            shap_strategy = ShapFactory.get_explainer(model_type)
            shap_values = shap_strategy.get_shap_values(classifier.model, X_sample)

            # Gestione output SHAP per classificazione binaria
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            elif len(np.shape(shap_values)) == 3:
                shap_values = shap_values[:, :, 1]

            vals=np.abs(shap_values).mean(0)

            feature_importance = pd.DataFrame(list(zip(range(len(vals)), vals)), columns=['feature_idx', 'importance'])
            feature_importance.sort_values(by='importance', ascending=False, inplace=True)

            feature_importance.to_csv(shap_cache_path, index=False)

        balanced_data = self.data_loader.balance_model(data)
        X_bal, y_bal = self.data_loader.prepare_features(balanced_data, train=True)
        metadata = balanced_data
        all_features = np.arange(X_bal.shape[1])

        feature_cols = balanced_data.drop(columns=self.config.data.cols_to_exclude, errors='ignore').columns
        save_embeddings = getattr(self.config.experiment, 'save_filtered_embeddings', False)

        n_features_list = self.config.experiment.n_features_to_remove_shap

        all_ablation_results = {}
        
        for n_features in n_features_list:
                
            top_features = feature_importance['feature_idx'].head(n_features).values
            print(f"\n>> Step di Tuning: Rimozione delle Top {n_features} feature proxy")

            if save_embeddings:
                features_list_path = f"{base_dir}/features_to_remove_{n_features}.csv"
                
                if os.path.exists(features_list_path) and not force_retrain:
                    pass 
                else:
                    cols_to_drop = feature_cols[top_features]
                    
                    features_df = pd.DataFrame({'feature_to_remove': cols_to_drop})
                    
                    os.makedirs(os.path.dirname(features_list_path), exist_ok=True)
                    features_df.to_csv(features_list_path, index=False)


            features_to_keep = np.setdiff1d(all_features, top_features)
            X_pruned = X_bal[:, features_to_keep]
            
            print(f"   Dimensione feature ridotta: da {X_bal.shape[1]} a {X_pruned.shape[1]}")
            
            sss = StratifiedShuffleSplit(n_splits=self.config.experiment.n_splits, 
                                        test_size=self.config.experiment.test_size, 
                                        random_state=self.config.experiment.random_seed)
            
            ablation_results = []
            
            for fold, (train_index, test_index) in enumerate(sss.split(X_pruned, y_bal)):
                X_train, X_test = X_pruned[train_index], X_pruned[test_index]
                y_train, y_test = y_bal[train_index], y_bal[test_index]
                meta_test = metadata.iloc[test_index]

                checkpoint_path = f"{base_dir}/fold_{fold}_{n_features}_pruned.joblib"
                
                metrics = self.run_fold_logic(X_train, y_train, X_test, y_test, meta_test, save_path=checkpoint_path)
                ablation_results.append(metrics)

                print(f"Fold {fold + 1}/{self.config.experiment.n_splits} completato con {n_features} feature rimosse.")
                
            all_ablation_results[n_features] = ablation_results
            
        return all_ablation_results
    
    #Task 5 
    def run_task_etnia(self, combined_data, fm_name):
        print(f"Avvio Task 5 (Domain Shift) per {fm_name}...")

        min_size = combined_data['dataset_label'].value_counts().min()

        balanced_data = combined_data.groupby('dataset_label', group_keys=False).apply(
            lambda x: x.sample(n=min_size, random_state=self.config.experiment.random_seed)
        ).reset_index(drop=True)

        cols_to_drop = getattr(self.config.data, 'cols_to_exclude', []) + ['dataset_label']
        X = balanced_data.drop(columns=cols_to_drop, errors='ignore').values
        y = balanced_data['dataset_label'].values

        sss = StratifiedShuffleSplit(n_splits=self.config.experiment.n_splits,
                                     test_size=self.config.experiment.test_size,
                                     random_state=self.config.experiment.random_seed)
        
        accuracies, f1_scores, aucs = [], [], []
        
        base_dir = f"checkpoints/{self.config.experiment.model_type}/Domain_Shift_{fm_name}/task5"
        os.makedirs(base_dir, exist_ok=True)
        
        for fold, (train_index, test_index) in enumerate(sss.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            checkpoint_path = f"{base_dir}/fold_{fold}.joblib"

            classifier = self._get_classifier()
            classifier.train(X_train_scaled, y_train, model_save_path=checkpoint_path)

            preds = classifier.predict(X_test_scaled)
            proba = classifier.predict_proba(X_test_scaled)[:, 1]
            accuracies.append(accuracy_score(y_test, preds))
            f1_scores.append(f1_score(y_test, preds))
            aucs.append(roc_auc_score(y_test, proba))
        
        return {
            'Accuracy': {'Med': np.median(accuracies), 'Q1': np.percentile(accuracies, 25), 'Q3': np.percentile(accuracies, 75)},
            'F1_Score': {'Med': np.median(f1_scores), 'Q1': np.percentile(f1_scores, 25), 'Q3': np.percentile(f1_scores, 75)},
            'AUC': {'Med': np.median(aucs), 'Q1': np.percentile(aucs, 25), 'Q3': np.percentile(aucs, 75)}
        }
    
    #Task 6 
    def run_correlation_analysis(self, data, fm_name, ds_name):
        print("Avvio Task 6: Analisi Correlazione tra Feature e Attributo di Bias...")

        features = data.drop(columns=self.config.data.cols_to_exclude, errors='ignore')
        corr_matrix = features.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        threshold = self.config.experiment.correlation_threshold
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]

        total_features = len(features.columns)
        num_redundant = len(to_drop)

        base_dir = f"results/correlation_analysis/{fm_name}/{ds_name}"
        os.makedirs(base_dir, exist_ok=True)

        report_path = f"{base_dir}/redundancy_report.txt"
        with open(report_path, "w") as f:
            f.write(f"REPORT RIDONDANZA EMBEDDING\n")
            f.write(f"Foundation Model: {fm_name}\nDataset: {ds_name}\n")
            f.write(f"Soglia: {threshold}\n{'-'*30}\n")
            f.write(f"Totale Feature: {total_features}\n")
            f.write(f"Feature Ridondanti da scartare: {num_redundant}\n\n")
            f.write("Elenco feature:\n" + ", ".join(to_drop))
            
        try:
            max_corrs = upper_tri.max()
            plt.figure(figsize=(10, 6))
            sns.histplot(max_corrs.dropna(), bins=50, kde=True, color='royalblue')
            plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Soglia Critica ({threshold})')
            plt.title(f'Ridondanza Embedding: {fm_name} | {ds_name}')
            plt.xlabel('Correlazione Massima')
            plt.ylabel('Numero di Feature')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{base_dir}/correlation_histogram.png", dpi=300)
            plt.close()
        except Exception as e:
            print(f"   [WARN] Errore nel grafico: {e}")

        return to_drop
 
    #Task 7   
    def run_shap_efficiency_analysis(self, data):
        print("Avvio Task 7: Analisi Efficienza SHAP...")

        model_type = self.config.experiment.model_type
        ds_name = self.config.data.current_dataset
        target = self.config.experiment.target_column[0]
        bias = self.config.experiment.bias_attribute[0]

        base_dir = f"checkpoints/{model_type}/{ds_name}/{target}_{bias}/efficiency_ablation"
        os.makedirs(base_dir, exist_ok=True)

        X, y = self.data_loader.prepare_features(data, train=True)
        force_retrain = getattr(self.config.experiment, 'force_retrain', False)

        shap_cache_path = f"{base_dir}/target_feature_importance.csv"

        if os.path.exists(shap_cache_path) and not force_retrain:
            print(f"   [INFO] Caricamento classifica feature dalla cache: {shap_cache_path}")
            feature_importance = pd.read_csv(shap_cache_path)
        else:
            print("   [INFO] Calcolo SHAP sulla Target Column per determinare l'importanza clinica...")
            classifier_detector = self._get_classifier()
            classifier_detector.train(X, y)

            shap_strategy = ShapFactory.get_explainer(model_type)

            sample_size = self.config.experiment.sample_size_shap
            X_sample = X[:sample_size] if sample_size < len(X) else X
            
            shap_values = shap_strategy.get_shap_values(classifier_detector.model, X_sample)

            if isinstance(shap_values, list): shap_values = shap_values[1]
            elif len(np.shape(shap_values)) == 3: shap_values = shap_values[:, :, 1]

            vals = np.abs(shap_values).mean(0)
            feature_importance = pd.DataFrame(list(zip(range(len(vals)), vals)), columns=['feature_idx', 'importance'])

            feature_importance.sort_values(by='importance', ascending=False, inplace=True)
            feature_importance.to_csv(shap_cache_path, index=False)
        
        balanced_data = self.data_loader.balance_model(data)
        X_bal, y_bal = self.data_loader.prepare_features(balanced_data, train=True)
        all_features_idxs = np.arange(X_bal.shape[1])
        feature_cols_names = balanced_data.drop(columns=self.config.data.cols_to_exclude, errors='ignore').columns

        n_to_remove_list = self.config.experiment.n_features_to_remove_shap_efficiency
        all_efficiency_results = {}

        if self.config.experiment.random_baseline:
            np.random.seed(self.config.experiment.random_seed)
            np.random.shuffle(feature_importance['feature_idx'].values)
            base_dir = f"checkpoints/{model_type}/{ds_name}/{target}_{bias}/efficiency_ablation/random_baseline"
            os.makedirs(base_dir, exist_ok=True)

        for n_remove in n_to_remove_list:
            bottom_features = feature_importance['feature_idx'].tail(n_remove).values
            features_to_keep = np.setdiff1d(all_features_idxs, bottom_features)

            print(f"\n>> Step: Test con {len(features_to_keep)} feature (Rimosse {n_remove} meno rilevanti)")

            removed_names_path = f"{base_dir}/removed_features_{n_remove}.csv"
            if not os.path.exists(removed_names_path) or force_retrain:
                pd.DataFrame({'removed_feature': feature_cols_names[bottom_features]}).to_csv(removed_names_path, index=False)

            X_pruned = X_bal[:, features_to_keep]
            
            sss = StratifiedShuffleSplit(n_splits=self.config.experiment.n_splits, 
                                        test_size=self.config.experiment.test_size, 
                                        random_state=self.config.experiment.random_seed)
            
            step_results = []
            for fold, (train_index, test_index) in enumerate(sss.split(X_pruned, y_bal)):
                X_train, X_test = X_pruned[train_index], X_pruned[test_index]
                y_train, y_test = y_bal[train_index], y_bal[test_index]
                meta_test = balanced_data.iloc[test_index]

                checkpoint_path = f"{base_dir}/fold_{fold}_{n_remove}_removed.joblib"
                
                # run_fold_logic gestisce internamente il caricamento se il file esiste
                metrics = self.run_fold_logic(X_train, y_train, X_test, y_test, meta_test, save_path=checkpoint_path)
                step_results.append(metrics)
                
            all_efficiency_results[n_remove] = step_results
            
        return all_efficiency_results
    

    def compare_features(self, task4_n, task7_n):
        ds_name = self.config.data.current_dataset
        model_type = self.config.experiment.model_type
        target = self.config.experiment.target_column[0] if isinstance(self.config.experiment.target_column, list) else self.config.experiment.target_column
        bias = self.config.experiment.bias_attribute[0] if isinstance(self.config.experiment.bias_attribute, list) else self.config.experiment.bias_attribute

        current_emb_path = None
        for fm in self.config.data.foundation_models:
            datasets_list = fm.get('datasets', [])
            for ds in datasets_list:
                if ds.get('name') == ds_name:
                    current_emb_path = ds.get('embedding_path')
                    break
            if current_emb_path: break

        if not current_emb_path:
            print(f" [!] Path embedding non trovato per {ds_name}")
            return set()

        path_t4 = f"checkpoints/{model_type}/{ds_name}/{target}_{bias}/shap_ablation/features_to_remove_{task4_n}.csv"
        path_t7 = f"checkpoints/{model_type}/{ds_name}/{target}_{bias}/efficiency_ablation/removed_features_{task7_n}.csv"
        
        try:
            
            df_emb = pd.read_csv(current_emb_path)
            all_features = set([c for c in df_emb.columns if c not in self.config.data.cols_to_exclude])
            
            feat_t4_proxies = set(pd.read_csv(path_t4)['feature_to_remove'].tolist())
            
            feat_t7_removed = set(pd.read_csv(path_t7)['removed_feature'].tolist())
            
            feat_t7_kept = all_features - feat_t7_removed
            
            intersection = feat_t4_proxies.intersection(feat_t7_kept)
            
            print(f" ANALISI DEL CONFLITTO: {self.config.data.current_dataset}")
            print(f" Feature totali embedding: {len(all_features)}")
            print(f" Feature rimosse embedding: {len(feat_t7_removed)}")
            print(f" Feature Proxy (Sesso): {len(feat_t4_proxies)}")
            print(f" Feature Core Clinico (AF): {len(feat_t7_kept)-len(feat_t4_proxies)+len(intersection)}")
            print(f" --------------------------------------")
            print(f" FEATURE IN CONFLITTO: {len(intersection)}")
            
            return intersection

        except Exception as e:
            print(f" Errore nel confronto: {e}")
            return set()
        
    #Task 8 
    def run_overlap_analysis(self, data):
        print("\n" + "="*50)
        print("AVVIO TASK 8: MITIGAZIONE CONFLITTI BIAS-CLINICA")
        print("="*50)

        ds_name = self.config.data.current_dataset
        model_type = self.config.experiment.model_type
        if not hasattr(self.config.experiment, ds_name):
            print(f"Parametri 'best' non trovati per {ds_name}. Task 8 annullato.")
            return None
        
        ds_config = getattr(self.config.experiment, ds_name)

        if model_type in ds_config:
            print(f"   [INFO] Caricamento parametri 'best' per {ds_name} - {model_type}")
            model_params = ds_config[model_type]
            best_n_t4 = model_params['best_t4']
            best_n_t7 = model_params['best_t7']
        else:
            print(f" [!] Parametri per {model_type} non trovati in {ds_name}. Task 8 annullato.")
            return None

        conflict_features = self.compare_features(
            task4_n=best_n_t4,
            task7_n=best_n_t7
        )

        if not conflict_features:
            print("Nessuna feature in conflitto trovata. Inutile procedere con l'ablazione.")
            return None
        
        target = self.config.experiment.target_column[0]
        bias = self.config.experiment.bias_attribute[0]

        path_t4 = f"checkpoints/{model_type}/{ds_name}/{target}_{bias}/shap_ablation/features_to_remove_{best_n_t4}.csv"
        path_t7 = f"checkpoints/{model_type}/{ds_name}/{target}_{bias}/efficiency_ablation/removed_features_{best_n_t7}.csv"
        
        feat_t4_proxies = set(pd.read_csv(path_t4)['feature_to_remove'].tolist())
        feat_t7_removed = set(pd.read_csv(path_t7)['removed_feature'].tolist())
        
        all_bad_features = feat_t4_proxies.union(feat_t7_removed)

        balanced_data = self.data_loader.balance_model(data)
        X_bal, y_bal = self.data_loader.prepare_features(balanced_data, train=True)
        
        # Mappo i nomi delle colonne per trovare gli indici da rimuovere
        feature_cols = balanced_data.drop(columns=self.config.data.cols_to_exclude, errors='ignore').columns
        bad_indices = [list(feature_cols).index(f) for f in all_bad_features if f in feature_cols]
        
        all_indices = np.arange(X_bal.shape[1])
        features_to_keep = np.setdiff1d(all_indices, bad_indices)

        X_pruned = X_bal[:, features_to_keep]
        print(f" Rimosse {len(bad_indices)} feature in conflitto. Nuova dimensione: {X_pruned.shape[1]}")

        sss = StratifiedShuffleSplit(n_splits=self.config.experiment.n_splits, 
                                    test_size=self.config.experiment.test_size, 
                                    random_state=self.config.experiment.random_seed)
        
        results = []
        for fold, (train_index, test_index) in enumerate(sss.split(X_pruned, y_bal)):
            X_train, X_test = X_pruned[train_index], X_pruned[test_index]
            y_train, y_test = y_bal[train_index], y_bal[test_index]
            meta_test = balanced_data.iloc[test_index]

            target = self.config.experiment.target_column[0]
            bias = self.config.experiment.bias_attribute[0]
            model_type = self.config.experiment.model_type
            save_dir = f"checkpoints/{model_type}/{ds_name}/{target}_{bias}/overlap_analysis"
            os.makedirs(save_dir, exist_ok=True)
            
            metrics = self.run_fold_logic(X_train, y_train, X_test, y_test, meta_test, 
                                        save_path=f"{save_dir}/fold_{fold}.joblib")
            results.append(metrics)
            
        return results
    
    #Task 9
    def run_bias_audit(self, data):
        print("\n" + "="*50)
        print("AVVIO TASK 9: AUDIT SUL BIAS")
        print(f"Dataset: {self.config.data.current_dataset}")
        print("="*50)

        ds_name = self.config.data.current_dataset
        model_type = self.config.experiment.model_type
        if not hasattr(self.config.experiment, ds_name):
            print(f"Parametri 'best' non trovati per {ds_name}. Salto.")
            return None
        
        ds_config = getattr(self.config.experiment, ds_name)

        if model_type in ds_config:
            #print(f"Caricamento parametri 'best' per {ds_name} - {model_type}")
            model_params = ds_config[model_type]
            best_n_t4 = model_params['best_t4']
            best_n_t7 = model_params['best_t7']
        else:
            #print(f" Parametri per {model_type} non trovati in {ds_name}. Task 8 annullato.")
            return None

        conflict_features = self.compare_features(
            task4_n=best_n_t4,
            task7_n=best_n_t7
        )

        target = self.config.experiment.target_column[0]
        bias_col = self.config.experiment.bias_attribute[0]

        path_t4 = f"checkpoints/{model_type}/{ds_name}/{target}_{bias_col}/shap_ablation/features_to_remove_{best_n_t4}.csv"
        path_t7 = f"checkpoints/{model_type}/{ds_name}/{target}_{bias_col}/efficiency_ablation/removed_features_{best_n_t7}.csv"
        
        feat_t4_proxies = set(pd.read_csv(path_t4)['feature_to_remove'].tolist())
        feat_t7_removed = set(pd.read_csv(path_t7)['removed_feature'].tolist())
        
        all_bad_features = feat_t4_proxies.union(feat_t7_removed)

        balanced_data = self.data_loader.balance_model(data)
        X_full, _ = self.data_loader.prepare_features(balanced_data, train=True)
        y_bias = balanced_data[bias_col].values 
        
        feature_cols = balanced_data.drop(columns=self.config.data.cols_to_exclude, errors='ignore').columns
        bad_indices = [list(feature_cols).index(f) for f in all_bad_features if f in feature_cols]
        all_indices = np.arange(X_full.shape[1])
        features_to_keep = np.setdiff1d(all_indices, bad_indices)

        X_clean = X_full[:, features_to_keep]
        #print(f" su {X_clean.shape[1]} feature superstiti.")

        model_type = self.config.experiment.model_type
        target_col = self.config.experiment.target_column[0]
        save_dir = f"checkpoints/{model_type}/{ds_name}/{target_col}_{bias_col}/task9_audit"
        os.makedirs(save_dir, exist_ok=True)

        sss = StratifiedShuffleSplit(n_splits=self.config.experiment.n_splits, 
                                    test_size=self.config.experiment.test_size, 
                                    random_state=self.config.experiment.random_seed)
        
        accuracies, f1_scores, aucs = [], [], []

        for fold, (train_index, test_index) in enumerate(sss.split(X_clean, y_bias)):
            X_train, X_test = X_clean[train_index], X_clean[test_index]
            y_train, y_test = y_bias[train_index], y_bias[test_index]

            checkpoint_path = f"{save_dir}/fold_{fold}.joblib"

            classifier = self._get_classifier()
            classifier.train(X_train, y_train, model_save_path=checkpoint_path) 
            
            preds = classifier.predict(X_test)
            proba = classifier.predict_proba(X_test)[:, 1]

            accuracies.append(accuracy_score(y_test, preds))
            f1_scores.append(f1_score(y_test, preds))
            aucs.append(roc_auc_score(y_test, proba))
            
        return {
            'Accuracy': {
                'Med': np.median(accuracies), 
                'Q1': np.percentile(accuracies, 25), 
                'Q3': np.percentile(accuracies, 75)
            },
            'F1_Score': {
                'Med': np.median(f1_scores), 
                'Q1': np.percentile(f1_scores, 25), 
                'Q3': np.percentile(f1_scores, 75)
            },
            'AUC': {
                'Med': np.median(aucs), 
                'Q1': np.percentile(aucs, 25), 
                'Q3': np.percentile(aucs, 75)
            }
        }
    

    def run_debiasing_post_task7(self, data):
        print("\n" + "="*50)
        print("AVVIO TASK 10: DE-BIASING POST-TASK 7 (0, 2, 4, 8, 16...)")
        print("="*50)

        ds_name = self.config.data.current_dataset
        model_type = self.config.experiment.model_type
        target = self.config.experiment.target_column[0]
        bias = self.config.experiment.bias_attribute[0]

        base_dir = f"checkpoints/{model_type}/{ds_name}/{target}_{bias}/debias_post_task7"
        os.makedirs(base_dir, exist_ok=True)

        ds_config = getattr(self.config.experiment, ds_name, {})
        if model_type not in ds_config:
            print(f"Parametri best non trovati per {ds_name}. Salto.")
            return None
        
        best_t7 = ds_config[model_type]['best_t7']
        if self.config.experiment.random_baseline:
            path_t7 = f"checkpoints/{model_type}/{ds_name}/{target}_{bias}/efficiency_ablation/random_baseline/removed_features_{best_t7}.csv"
        else:        
            path_t7 = f"checkpoints/{model_type}/{ds_name}/{target}_{bias}/efficiency_ablation/removed_features_{best_t7}.csv"
        if not os.path.exists(path_t7):
            print(f"Manca il file CSV della Task 7 per {ds_name}.")
            return None
            
        feat_t7_removed = set(pd.read_csv(path_t7)['removed_feature'].tolist())

        balanced_data = self.data_loader.balance_model(data)
        X_full, _ = self.data_loader.prepare_features(balanced_data, train=True)
        feature_cols = balanced_data.drop(columns=self.config.data.cols_to_exclude, errors='ignore').columns

        t7_indices = [list(feature_cols).index(f) for f in feat_t7_removed if f in feature_cols]
        all_indices = np.arange(X_full.shape[1])
        features_to_keep_t7 = np.setdiff1d(all_indices, t7_indices)

        # Dataset base post-Task 7
        X_clinical = X_full[:, features_to_keep_t7]
        clinical_feature_names = feature_cols[features_to_keep_t7]
        y_af = balanced_data[target].values
        y_sex = balanced_data[bias].values

        num_clinical_features = X_clinical.shape[1]
        print(f" -> Feature cliniche superstiti (Post-T7): {num_clinical_features}")

        # SHAP per il sesso
        shap_cache_path = f"{base_dir}/shap_feature_importance.csv"
        force_retrain = getattr(self.config.experiment, 'force_retrain', False)

        if os.path.exists(shap_cache_path) and not force_retrain:
            print(f"   [INFO] Caricamento classifica feature dalla cache: {shap_cache_path}")
            feature_importance = pd.read_csv(shap_cache_path)
        else:
            classifier_shap = self._get_classifier()
            shap_model_path = f"{base_dir}/shap_detector_model.joblib"
            classifier_shap.train(X_clinical, y_sex, model_save_path=shap_model_path) 

            shap_strategy = ShapFactory.get_explainer(model_type)
            sample_size = self.config.experiment.sample_size_shap
            X_sample = X_clinical[:sample_size] if sample_size < len(X_clinical) else X_clinical

            shap_values = shap_strategy.get_shap_values(classifier_shap.model, X_sample)
            if isinstance(shap_values, list): shap_values = shap_values[1]
            elif len(np.shape(shap_values)) == 3: shap_values = shap_values[:, :, 1]

            vals = np.abs(shap_values).mean(0)
            feature_importance = pd.DataFrame({
                'local_idx': range(len(vals)), 
                'feature_name': clinical_feature_names,
                'importance': vals
            })
            feature_importance.sort_values(by='importance', ascending=False, inplace=True)
            feature_importance.to_csv(shap_cache_path, index=False)

        # 3. Sequenza esponenziale
        removal_steps = [0]
        k = 2
        while k < num_clinical_features:
            removal_steps.append(k)
            k *= 2

        sss = StratifiedShuffleSplit(n_splits=self.config.experiment.n_splits, test_size=self.config.experiment.test_size, random_state=self.config.experiment.random_seed)

        all_step_results = {}

        if self.config.experiment.random_baseline:
            np.random.seed(self.config.experiment.random_seed)
            np.random.shuffle(feature_importance['local_idx'].values)
            base_dir = f"checkpoints/{model_type}/{ds_name}/{target}_{bias}/debias_post_task7/random_baseline"
            os.makedirs(base_dir, exist_ok=True)

        for n_remove in removal_steps:
            print(f"   >> Test con {n_remove} feature proxy rimosse...")
            top_proxy_local_idxs = feature_importance['local_idx'].head(n_remove).values

            if n_remove > 0:
                removed_feature_names = feature_importance['feature_name'].head(n_remove).values
                df_removed = pd.DataFrame({'removed_feature': removed_feature_names})
                removed_csv_path = f"{base_dir}/removed_features_step_{n_remove}.csv"
                df_removed.to_csv(removed_csv_path, index=False)

            clinical_indices = np.arange(num_clinical_features)
            features_to_keep_now = np.setdiff1d(clinical_indices, top_proxy_local_idxs)

            X_step = X_clinical[:, features_to_keep_now]

            af_metrics_list = []
            audit_acc_list = []

            for fold, (train_idx, test_idx) in enumerate(sss.split(X_step, y_af)):
                X_train, X_test = X_step[train_idx], X_step[test_idx]
                y_train_af, y_test_af = y_af[train_idx], y_af[test_idx]
                y_train_sex, y_test_sex = y_sex[train_idx], y_sex[test_idx]
                meta_test = balanced_data.iloc[test_idx]

                checkpoint_af = f"{base_dir}/fold_{fold}_{n_remove}_AF.joblib"
                
                metrics = self.run_fold_logic(X_train, y_train_af, X_test, y_test_af, meta_test, save_path=checkpoint_af)
                af_metrics_list.append(metrics)

                checkpoint_sex = f"{base_dir}/fold_{fold}_{n_remove}_SEX.joblib"

                clf_sex = self._get_classifier()
                clf_sex.train(X_train, y_train_sex, model_save_path=checkpoint_sex)
                preds_sex = clf_sex.predict(X_test)
                audit_acc_list.append(f1_score(y_test_sex, preds_sex, average='macro'))

            all_step_results[n_remove] = {
                'af_results': af_metrics_list,  
                'audit_acc': np.median(audit_acc_list)
            }

        return all_step_results
    
    def analyze_feature_overlap_from_csv(self, ds_name_1, ds_name_2):
        print(f"\n{'='*70}")
        print(f"PARTE 3: ANALISI SOVRAPPOSIZIONE FEATURE ({ds_name_1} vs {ds_name_2})")
        print(f"{'='*70}")

        model_type = self.config.experiment.model_type
        target = self.config.experiment.target_column[0]
        bias = self.config.experiment.bias_attribute[0]

        def get_surviving_features(ds_name):
            ds_config = getattr(self.config.experiment, ds_name, {})
            model_config = ds_config.get(model_type, {})
            best_t7 = model_config.get('best_t7', 0)
            best_t4 = model_config.get('best_t4_post_t7', 0)

            all_features = set([f"feature_{i}" for i in range(768)]) 

            path_t7 = f"checkpoints/{model_type}/{ds_name}/{target}_{bias}/efficiency_ablation/removed_features_{best_t7}.csv"
            removed_t7 = set()
            try:
                removed_t7 = set(pd.read_csv(path_t7)['removed_feature'].astype(str).tolist())
            except FileNotFoundError:
                print(f"   [ATTENZIONE] CSV Task 7 non trovato per {ds_name}: {path_t7}")

            path_debias = f"checkpoints/{model_type}/{ds_name}/{target}_{bias}/debias_post_task7/removed_features_step_{best_t4}.csv"
            removed_t8 = set()
            try:
                removed_t8 = set(pd.read_csv(path_debias)['removed_feature'].astype(str).tolist())
            except FileNotFoundError:
                print(f"   [ATTENZIONE] CSV Debiasing non trovato per {ds_name}: {path_debias}")

            surviving = all_features - removed_t7 - removed_t8
            return list(surviving)

        surviving_1 = get_surviving_features(ds_name_1)
        surviving_2 = get_surviving_features(ds_name_2)

        intersezione = np.intersect1d(surviving_1, surviving_2)
        unione = np.union1d(surviving_1, surviving_2)

        print(f"\n   [RISULTATI SOVRAPPOSIZIONE {model_type.upper()}]")
        print(f"   - Feature rimaste filtrate solo su {ds_name_1}: {len(surviving_1)}")
        print(f"   - Feature rimaste filtrate solo su {ds_name_2}: {len(surviving_2)}")
        print(f"   - Feature IN COMUNE (Intersezione): {len(intersezione)}")
        print(f"   - Feature TOTALI combinate (Unione):  {len(unione)}")

        out_dir = f"results/cross_dataset/{model_type}"
        os.makedirs(out_dir, exist_ok=True)
        
        pd.DataFrame({'Common_Features_Intersection': intersezione}).to_csv(f"{out_dir}/{ds_name_1}_vs_{ds_name_2}_intersection.csv", index=False)
        pd.DataFrame({'All_Features_Union': unione}).to_csv(f"{out_dir}/{ds_name_1}_vs_{ds_name_2}_union.csv", index=False)
        
        print(f"   -> Liste salvate con successo in: {out_dir}/")
        
        return intersezione, unione
    

    def run_cross_dataset_base(self, data_train, data_test, train_name, test_name):
        print(f"\n   -> [CROSS-DATASET] PARTE 1: BASE (Full Embedding)")
        print(f"      Train: [{train_name}] | Test: [{test_name}]")

        balanced_train = self.data_loader.balance_model(data_train)
        balanced_test = self.data_loader.balance_model(data_test)

        X_train, y_train = self.data_loader.prepare_features(balanced_train, train=True)
        X_test, y_test = self.data_loader.prepare_features(balanced_test, train=False)
        
        meta_test = balanced_test 

        print(f"      [INFO] Campioni Train: {X_train.shape[0]} | Campioni Test: {X_test.shape[0]}")
        print(f"      [INFO] Feature utilizzate: {X_train.shape[1]}")

        model_type = self.config.experiment.model_type
        save_dir = f"checkpoints/{model_type}/cross_dataset/parte1_base/{train_name}_to_{test_name}"
        os.makedirs(save_dir, exist_ok=True)
        model_path = f"{save_dir}/model.joblib"

        print("\n      [DEBUG] --- ISPEZIONE MATRICI PRE-TRAINING ---")
        
        # Controllo valori nulli o infiniti
        nan_train = np.isnan(X_train).sum()
        nan_test = np.isnan(X_test).sum()
        inf_train = np.isinf(X_train).sum()
        inf_test = np.isinf(X_test).sum()
        if nan_train > 0 or nan_test > 0 or inf_train > 0 or inf_test > 0:
            print(f"      [!] ALLARME: Trovati NaN (Train:{nan_train}, Test:{nan_test}) o Inf (Train:{inf_train}, Test:{inf_test})")
            
        # Controllo range dei valori 
        print(f"      [DEBUG] X_train -> Min: {np.min(X_train):.4f} | Max: {np.max(X_train):.4f} | Media: {np.mean(X_train):.4f}")
        print(f"      [DEBUG] X_test  -> Min: {np.min(X_test):.4f} | Max: {np.max(X_test):.4f} | Media: {np.mean(X_test):.4f}")
        
        # Controllo bilanciamento etichette (y)
        unique_y_train, counts_y_train = np.unique(y_train, return_counts=True)
        unique_y_test, counts_y_test = np.unique(y_test, return_counts=True)
        print(f"      [DEBUG] Distribuzione y_train: {dict(zip(unique_y_train, counts_y_train))}")
        print(f"      [DEBUG] Distribuzione y_test:  {dict(zip(unique_y_test, counts_y_test))}")
        
        # Controllo identità del sesso nei test set per il calcolo delle metriche
        bias_col = self.config.experiment.bias_attribute[0]
        unique_sex, counts_sex = np.unique(meta_test[bias_col].dropna(), return_counts=True)
        print(f"      [DEBUG] Sottopopolazioni in meta_test ({bias_col}): {dict(zip(unique_sex, counts_sex))}")
        print("      ----------------------------------------------\n")

        metrics = self.run_fold_logic(X_train, y_train, X_test, y_test, meta_test, save_path=model_path)

        metrics['Task'] = "CrossDataset (Parte 1: Base Full)"
        metrics['Dataset'] = f"Train:{train_name} -> Test:{test_name}"
        metrics['Model'] = model_type

        return [metrics]
    


    def run_cross_dataset_filtered(self, data_train, data_test, train_name, test_name):
        print(f"\n   -> [CROSS-DATASET] PARTE 2: FILTRATO (Maschera del Train)")
        print(f"      Train: [{train_name}] | Test: [{test_name}]")

        balanced_train = self.data_loader.balance_model(data_train)
        balanced_test = self.data_loader.balance_model(data_test)

        feature_cols = balanced_train.drop(columns=self.config.data.cols_to_exclude, errors='ignore').columns

        X_train, y_train = self.data_loader.prepare_features(balanced_train, train=True)
        X_test, y_test = self.data_loader.prepare_features(balanced_test, train=False)
        meta_test = balanced_test 

        model_type = self.config.experiment.model_type
        target = self.config.experiment.target_column[0]
        bias = self.config.experiment.bias_attribute[0]

        ds_config = getattr(self.config.experiment, train_name, {})
        model_config = ds_config.get(model_type, {})
        best_t7 = model_config.get('best_t7', 0)
        best_t4 = model_config.get('best_t4_post_t7', 0)

        all_features = set(feature_cols)

        path_t7 = f"checkpoints/{model_type}/{train_name}/{target}_{bias}/efficiency_ablation/removed_features_{best_t7}.csv"
        removed_t7 = set()
        try:
            removed_t7 = set(pd.read_csv(path_t7)['removed_feature'].astype(str).tolist())
        except FileNotFoundError:
            print(f"   [ATTENZIONE] CSV Task 7 non trovato per {train_name}: {path_t7}")

        path_debias = f"checkpoints/{model_type}/{train_name}/{target}_{bias}/debias_post_task7/removed_features_step_{best_t4}.csv"
        removed_t8 = set()
        try:
            removed_t8 = set(pd.read_csv(path_debias)['removed_feature'].astype(str).tolist())
        except FileNotFoundError:
            print(f"   [ATTENZIONE] CSV Debiasing non trovato per {train_name}: {path_debias}")

        surviving = all_features - removed_t7 - removed_t8

        indices_to_keep = sorted([list(feature_cols).index(f) for f in surviving if f in feature_cols])
        
        X_train = X_train[:, indices_to_keep]
        X_test = X_test[:, indices_to_keep]

        print(f"      [INFO] Feature utilizzate (Sopravvissute su {train_name}): {X_train.shape[1]}")

        save_dir = f"checkpoints/{model_type}/cross_dataset/parte2_filtered/{train_name}_to_{test_name}"
        os.makedirs(save_dir, exist_ok=True)
        model_path = f"{save_dir}/model.joblib"

        print("\n      [DEBUG] --- ISPEZIONE MATRICI PRE-TRAINING ---")
        
        # Controllo valori nulli o infiniti
        nan_train = np.isnan(X_train).sum()
        nan_test = np.isnan(X_test).sum()
        inf_train = np.isinf(X_train).sum()
        inf_test = np.isinf(X_test).sum()
        if nan_train > 0 or nan_test > 0 or inf_train > 0 or inf_test > 0:
            print(f"      [!] ALLARME: Trovati NaN (Train:{nan_train}, Test:{nan_test}) o Inf (Train:{inf_train}, Test:{inf_test})")
            
        # Controllo range dei valori (Scala)
        print(f"      [DEBUG] X_train -> Min: {np.min(X_train):.4f} | Max: {np.max(X_train):.4f} | Media: {np.mean(X_train):.4f}")
        print(f"      [DEBUG] X_test  -> Min: {np.min(X_test):.4f} | Max: {np.max(X_test):.4f} | Media: {np.mean(X_test):.4f}")
        
        # Controllo bilanciamento etichette (y)
        unique_y_train, counts_y_train = np.unique(y_train, return_counts=True)
        unique_y_test, counts_y_test = np.unique(y_test, return_counts=True)
        print(f"      [DEBUG] Distribuzione y_train: {dict(zip(unique_y_train, counts_y_train))}")
        print(f"      [DEBUG] Distribuzione y_test:  {dict(zip(unique_y_test, counts_y_test))}")
        
        # Controllo identità del sesso nei test set per il calcolo delle metriche
        bias_col = self.config.experiment.bias_attribute[0]
        unique_sex, counts_sex = np.unique(meta_test[bias_col].dropna(), return_counts=True)
        print(f"      [DEBUG] Sottopopolazioni in meta_test ({bias_col}): {dict(zip(unique_sex, counts_sex))}")
        print("      ----------------------------------------------\n")

        metrics = self.run_fold_logic(X_train, y_train, X_test, y_test, meta_test, save_path=model_path)

        metrics['Task'] = "CrossDataset (Parte 2: Filtrato)"
        metrics['Dataset'] = f"Train:{train_name} -> Test:{test_name}"
        metrics['Model'] = model_type

        return [metrics]
    


    def run_cross_dataset_union(self, data_train, data_test, train_name, test_name, union_features):
        print(f"\n   -> [CROSS-DATASET] PARTE 3: UNIONE (Feature Universali)")
        print(f"      Train: [{train_name}] | Test: [{test_name}]")

        balanced_train = self.data_loader.balance_model(data_train)
        balanced_test = self.data_loader.balance_model(data_test)

        feature_cols = balanced_train.drop(columns=self.config.data.cols_to_exclude, errors='ignore').columns

        X_train, y_train = self.data_loader.prepare_features(balanced_train, train=True)
        X_test, y_test = self.data_loader.prepare_features(balanced_test, train=False)
        meta_test = balanced_test 

        indices_to_keep = sorted([list(feature_cols).index(f) for f in union_features if f in feature_cols])
        
        X_train = X_train[:, indices_to_keep]
        X_test = X_test[:, indices_to_keep]

        print(f"      [INFO] Feature utilizzate (Unione Universale): {X_train.shape[1]}")

        model_type = self.config.experiment.model_type
        save_dir = f"checkpoints/{model_type}/cross_dataset/parte3_union/{train_name}_to_{test_name}"
        os.makedirs(save_dir, exist_ok=True)
        model_path = f"{save_dir}/model.joblib"

        print("\n      [DEBUG] --- ISPEZIONE MATRICI PRE-TRAINING ---")
        
        # Controllo valori nulli o infiniti
        nan_train = np.isnan(X_train).sum()
        nan_test = np.isnan(X_test).sum()
        inf_train = np.isinf(X_train).sum()
        inf_test = np.isinf(X_test).sum()
        if nan_train > 0 or nan_test > 0 or inf_train > 0 or inf_test > 0:
            print(f"      [!] ALLARME: Trovati NaN (Train:{nan_train}, Test:{nan_test}) o Inf (Train:{inf_train}, Test:{inf_test})")
            
        # Controllo range dei valori (Scala)
        print(f"      [DEBUG] X_train -> Min: {np.min(X_train):.4f} | Max: {np.max(X_train):.4f} | Media: {np.mean(X_train):.4f}")
        print(f"      [DEBUG] X_test  -> Min: {np.min(X_test):.4f} | Max: {np.max(X_test):.4f} | Media: {np.mean(X_test):.4f}")
        
        # Controllo bilanciamento etichette (y)
        unique_y_train, counts_y_train = np.unique(y_train, return_counts=True)
        unique_y_test, counts_y_test = np.unique(y_test, return_counts=True)
        print(f"      [DEBUG] Distribuzione y_train: {dict(zip(unique_y_train, counts_y_train))}")
        print(f"      [DEBUG] Distribuzione y_test:  {dict(zip(unique_y_test, counts_y_test))}")
        
        # Controllo identità del sesso nei test set per il calcolo delle metriche
        bias_col = self.config.experiment.bias_attribute[0]
        unique_sex, counts_sex = np.unique(meta_test[bias_col].dropna(), return_counts=True)
        print(f"      [DEBUG] Sottopopolazioni in meta_test ({bias_col}): {dict(zip(unique_sex, counts_sex))}")
        print("      ----------------------------------------------\n")

        metrics = self.run_fold_logic(X_train, y_train, X_test, y_test, meta_test, save_path=model_path)

        metrics['Task'] = "CrossDataset (Parte 3: Unione)"
        metrics['Dataset'] = f"Train:{train_name} -> Test:{test_name}"
        metrics['Model'] = model_type

        return [metrics]