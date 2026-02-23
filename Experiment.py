from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

from ConfigLoader import ConfigLoader
from DataLoader import DataLoader
from MetricManager import MetricManager
from ShapExplainer import ShapFactory
from Classifier import LogisticRegressionClassifier, RandomForestClassifierModel
class Experiment:
    def __init__(self, config_path):
        self.config = ConfigLoader(config_path)
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
        mask_m = meta_test['sex'] == self.config.experiment.male_value
        mask_f = meta_test['sex'] == self.config.experiment.female_value

        X_test_m = X_test[mask_m]
        y_test_m = y_test[mask_m]

        X_test_f = X_test[mask_f]
        y_test_f = y_test[mask_f]

        preds_m = classifier.predict(X_test_m)
        preds_f = classifier.predict(X_test_f)

        # Probabilità per calcolo AUROC
        proba_m = classifier.predict_proba(X_test_m)[:, 1]
        proba_f = classifier.predict_proba(X_test_f)[:, 1]

        # Gruppi per calcolo metriche
        group_a = {'true': y_test_m, 'pred': preds_m, 'proba': proba_m}
        group_b = {'true': y_test_f, 'pred': preds_f, 'proba': proba_f}

        results = self.metric_manager.compute_metrics(group_a, group_b)
        return results
    

    # Caso Base: Addestramento su dati bilanciati di entrambi i sessi e inferenza su entrambi i sessi separatamente.
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

           checkpoint_path = f"checkpoints/{self.config.experiment.model_type}/base_case/fold_{fold}.joblib"

           metrics = self.run_fold_logic(X_train, y_train, X_test, y_test, meta_test, save_path=checkpoint_path)
           all_results.append(metrics)

           #print(f"Fold {fold + 1}/{self.config.experiment.n_splits} completato.") 

       #print(f"Esperimento completato su {len(all_results)} iterazioni.")

       return all_results
    
    #Caso Estremo: Addestramento solo su un sesso (es. maschi) e inferenza su entrambi i sessi separatamente.
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

            checkpoint_path = f"checkpoints/{self.config.experiment.model_type}/extreme_case/fold_{fold}.joblib"

            metrics = self.run_fold_logic(X_train_male, y_train_male, X_test, y_test, meta_test, save_path=checkpoint_path)
            all_results.append(metrics)

            #print(f"Fold {fold + 1}/{self.config.experiment.n_splits} completato.")
        
        #print(f"Esperimento completato su {len(all_results)} iterazioni.")
        return all_results


    # Caso Controfattuale: Misura la percentuale di campioni il cui risultato predetto cambia quando si inverte l'attributo di bias. 
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
            checkpoint_path = f"checkpoints/{self.config.experiment.model_type}/counterfactual_case/fold_{fold}.joblib"
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
    


    def run_shap_analysis(self,data):
        print("--- Avvio Analisi SHAP e Rimozione Feature (Task 4) ---")

        X, _ = self.data_loader.prepare_features(data, train=True)
        y_sex = data["sex"].values
        model_type = self.config.experiment.model_type

        classifier = self._get_classifier()
        shap_model_path = f"checkpoints/{model_type}/shap_ablation/shap_detector_model.joblib"
        classifier.train(X, y_sex, model_save_path=shap_model_path)

        shap_cache_path = f"checkpoints/{model_type}/shap_ablation/feature_importance.csv"
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
                features_list_path = f"checkpoints/{model_type}/shap_ablation/features_to_remove_{n_features}.csv"
                
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

                checkpoint_path = f"checkpoints/{model_type}/shap_ablation/fold_{fold}_{n_features}_pruned.joblib"
                
                metrics = self.run_fold_logic(X_train, y_train, X_test, y_test, meta_test, save_path=checkpoint_path)
                ablation_results.append(metrics)
                
            all_ablation_results[n_features] = ablation_results
            
        return all_ablation_results