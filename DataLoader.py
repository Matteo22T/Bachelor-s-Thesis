import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



class DataLoader:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
    
    def load_data(self):
        embeddings = pd.read_csv(self.config.data.embeddings_path)
        metadata = pd.read_csv(self.config.data.metadata_path)
        return pd.merge(embeddings, metadata, on=self.config.data.join_key)
    
    def balance_model(self, data):
        strategy = getattr(self.config.experiment, 'balancing_strategy', None)
        target_col = self.config.experiment.target_column[0]
        bias_col = self.config.experiment.bias_attribute[0]

        if strategy == None:
            return data
        
        group_cols = []
        if strategy in ["target", "both"]: group_cols.append(target_col)
        if strategy in ["bias", "both"]: group_cols.append(bias_col)

        min_size = data.groupby(group_cols).size().min()

        balanced_data = data.groupby(group_cols, group_keys=False).apply(lambda x: x.sample(min_size, random_state=self.config.experiment.random_seed))
        return balanced_data.reset_index(drop=True)


    def prepare_features(self, data, train=True):
        target_col = self.config.experiment.target_column[0]
        metadata_to_concat = getattr(self.config.data, 'metadata_to_concatenate', [])
        X = data.drop(columns=(self.config.data.cols_to_exclude), errors='ignore')

        

        if train:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        if metadata_to_concat:
            X_meta = data[metadata_to_concat].values
            X_final = np.hstack((X_scaled, X_meta))
        else:
            X_final = X_scaled

        return X_final, data[target_col].values