from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from abc import ABC, abstractmethod
import os
import joblib

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class Classifier(ABC):
    def __init__(self, config):
        self.config = config
        self.model = self._build_model()

    @abstractmethod
    def _build_model(self):
        pass

    @abstractmethod
    def _get_params_grid(self):
        pass

    def train(self, X, y, model_save_path=None):

        force_retrain = getattr(self.config.experiment, 'force_retrain', False)

        if model_save_path and os.path.exists(model_save_path) and not force_retrain:
            self.model = joblib.load(model_save_path)
            return

        do_grid_search = getattr(self.config.grid_search, 'enabled', False)

        if do_grid_search:
            param_grid = self._get_params_grid()
            if param_grid:
                print(f"   [GridSearch] Ottimizzazione in corso... ", end="", flush=True)
                
                cv_splits = getattr(self.config.grid_search, 'cv_splits', 3)
                
                grid = GridSearchCV(estimator=self.model, 
                                    param_grid=param_grid, 
                                    cv=cv_splits, 
                                    scoring='accuracy',
                                    n_jobs=-1) 
                
                grid.fit(X, y)
                
                # Il modello diventa il "miglior stimatore" trovato
                self.model = grid.best_estimator_
                print(f"Fatto. Best params: {grid.best_params_}")
            else:
                self.model.fit(X, y)
        else:
            self.model.fit(X, y)
        
        if model_save_path:
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            joblib.dump(self.model, model_save_path)

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
class LogisticRegressionClassifier(Classifier):
    def _build_model(self):
        return LogisticRegression(
            C=0.01, 
            solver='liblinear', 
            penalty='l2', 
            max_iter=100, 
            random_state=self.config.experiment.random_seed
        )
    
    def _get_params_grid(self):
        return self.config.grid_search.logistic_regression
    
class RandomForestClassifierModel(Classifier):
    def _build_model(self):
        return RandomForestClassifier(
            n_estimators=200,      
            max_depth=20,          
            min_samples_split=5,   
            random_state=self.config.experiment.random_seed,
            n_jobs=-1
        )
    
    def _get_params_grid(self):
        return self.config.grid_search.random_forest