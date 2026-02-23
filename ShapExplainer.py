from abc import ABC, abstractmethod
import shap

class ShapExplainer(ABC):
    @abstractmethod
    def get_shap_values(self, model, X_sample):
        pass

class TreeShapExplainer(ShapExplainer):
    def get_shap_values(self, model, X_sample):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        return shap_values
    
class LinearShapExplainer(ShapExplainer):
    def get_shap_values(self, model, X_sample):
        explainer = shap.LinearExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)
        return shap_values
    
class ShapFactory:
    @staticmethod
    def get_explainer(model_type):
        if model_type == "random_forest":
            return TreeShapExplainer()
        elif model_type == "logistic_regression":
            return LinearShapExplainer()
        else:
            raise ValueError(f"SHAP non supportato per il modello: {model_type}")