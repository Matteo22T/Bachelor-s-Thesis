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

            train_male_mask = meta_train['sex'] == self.config.experiment.male_value

            X_train_male = X_train_full[train_male_mask]
            y_train_male = y_train_full[train_male_mask]

            ds_name = self.config.data.current_dataset
            target = self.config.experiment.target_column[0]
            bias = self.config.experiment.bias_attribute[0]

            base_dir = f"checkpoints/{self.config.experiment.model_type}/{ds_name}/{target}_{bias}/extreme_case"
            os.makedirs(base_dir, exist_ok=True)
            checkpoint_path = f"{base_dir}/fold_{fold}.joblib"

            metrics = self.run_fold_logic(X_train_male, y_train_male, X_test, y_test, meta_test, save_path=checkpoint_path)
            all_results.append(metrics)

            #print(f"Fold {fold + 1}/{self.config.experiment.n_splits} completato.")
        
        #print(f"Esperimento completato su {len(all_results)} iterazioni.")
        return all_results


    #Task 3 Caso Controfattuale: Misura la percentuale di campioni il cui risultato predetto cambia quando si inverte l'attributo di bias. 
    # Obiettivo: Valutare la robustezza del modello rispetto a cambiamenti nell'attributo di bias.
    def run_counterfactual_case(self, data):
        print("Avvio caso controfattuale...")

        original_meta_data = getattr(self.config.data, "metadata_to_concatenate", [])

        # Soluzione rapida per aggiungere i metadata da concatenare poichè se lo facessi dal config.yaml
        # influisce su gli altri esperimenti e non potrei runnarli tutti in sequenza
        temporarily_added = original_meta_data.copy()
        if 'sex' not in temporarily_added:
            temporarily_added.append('sex')
            self.config.data.metadata_to_concatenate = temporarily_added

        balanced_data = self.data_loader.balance_model(data)

        X, y = self.data_loader.prepare_features(balanced_data, train=True) 

        sex_col_index = X.shape[1] - 1

        sss = StratifiedShuffleSplit(n_splits=self.config.experiment.n_splits,
                                     test_size=self.config.experiment.test_size,
                                     random_state=self.config.experiment.random_seed)
        
        flips_count = []

        for fold, (train_index, test_index) in enumerate(sss.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            classifier = self._get_classifier()

            ds_name = self.config.data.current_dataset
            target = self.config.experiment.target_column[0]
            bias = self.config.experiment.bias_attribute[0]
            
            base_dir = f"checkpoints/{self.config.experiment.model_type}/{ds_name}/{target}_{bias}/counterfactual_case"
            os.makedirs(base_dir, exist_ok=True)
            checkpoint_path = f"{base_dir}/fold_{fold}.joblib"

            classifier.train(X_train, y_train, model_save_path=checkpoint_path)

            preds_original = classifier.predict(X_test)
            X_test_cf = X_test.copy()

            X_test_cf[:, sex_col_index] = 1 - X_test_cf[:, sex_col_index]

            preds_cf = classifier.predict(X_test_cf)

            flips = np.sum(preds_original != preds_cf)
            flips_rate = flips / len(preds_original)
            flips_count.append(flips_rate)

            #print(f"Fold {fold+1}: Flip Rate = {flips_rate*100:.2f}% ({flips}/{len(preds_original)} cambi)")

        self.config.data.metadata_to_concatenate = original_meta_data
       
        avg_flip_rate = np.mean(flips_count)
        print(f"\n{'='*10} RISULTATI TASK 3 {'='*10}")
        print(f"-> Counterfactual Flip Rate medio: {avg_flip_rate*100:.2f}%")
        return 
    

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
            print(f" Feature Core Clinico (AF): {len(feat_t7_kept)}")
            print(f" --------------------------------------")
            print(f" FEATURE IN CONFLITTO: {len(intersection)}")
            print(f" Impatto: {len(intersection)/len(feat_t4_proxies)*100:.2f}% delle proxy sono essenziali.")
            
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
        if not hasattr(self.config.experiment, ds_name):
            print(f"Parametri 'best' non trovati per {ds_name}. Task 8 annullato.")
            return None
        
        ds_config = getattr(self.config.experiment, ds_name)
        conflict_features = self.compare_features(
            task4_n=ds_config.get('best_t4'), 
            task7_n=ds_config.get('best_t7')
        )

        if not conflict_features:
            print("Nessuna feature in conflitto trovata. Inutile procedere con l'ablazione.")
            return None

        balanced_data = self.data_loader.balance_model(data)
        X_bal, y_bal = self.data_loader.prepare_features(balanced_data, train=True)
        
        # Mappo i nomi delle colonne per trovare gli indici da rimuovere
        feature_cols = balanced_data.drop(columns=self.config.data.cols_to_exclude, errors='ignore').columns
        conflict_indices = [list(feature_cols).index(f) for f in conflict_features]
        
        all_indices = np.arange(X_bal.shape[1])
        features_to_keep = np.setdiff1d(all_indices, conflict_indices)

        X_pruned = X_bal[:, features_to_keep]
        print(f" Rimosse {len(conflict_indices)} feature in conflitto. Nuova dimensione: {X_pruned.shape[1]}")

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
    

    def run_bias_audit(self, data):
        print("\n" + "="*50)
        print("AVVIO TASK 9: AUDIT SUL BIAS")
        print(f"Dataset: {self.config.data.current_dataset}")
        print("="*50)

        ds_name = self.config.data.current_dataset
        if not hasattr(self.config.experiment, ds_name):
            print(f"Parametri 'best' non trovati per {ds_name}. Salto.")
            return None
        
        ds_config = getattr(self.config.experiment, ds_name)
        conflict_features = self.compare_features(
            task4_n=ds_config.get('best_t4'), 
            task7_n=ds_config.get('best_t7')
        )

        bias_col = self.config.experiment.bias_attribute[0]
        balanced_data = self.data_loader.balance_model(data)
        X_full, _ = self.data_loader.prepare_features(balanced_data, train=True)
        y_bias = balanced_data[bias_col].values 
        
        feature_cols = balanced_data.drop(columns=self.config.data.cols_to_exclude, errors='ignore').columns
        conflict_indices = [list(feature_cols).index(f) for f in conflict_features]
        all_indices = np.arange(X_full.shape[1])
        features_to_keep = np.setdiff1d(all_indices, conflict_indices)

        X_clean = X_full[:, features_to_keep]
        print(f"Test di 'cecità' su {X_clean.shape[1]} feature superstiti.")

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