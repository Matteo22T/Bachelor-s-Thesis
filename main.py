import pandas as pd
import yaml
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"error, file not found: {config_path}")
        exit()

def load_data(embeddings_path, metadata_path, join_key):
    print("loading data...")
    df_emb = pd.read_csv(embeddings_path)
    df_meta = pd.read_csv(metadata_path)

    data = pd.merge(df_emb, df_meta, on=join_key)
    return data

def prepare_features(data, config):
    target_col = config['experiment']['target_column']
    id_col = config['data']['join_key']
    bias_col = config['experiment']['bias_attribute']

    y = data[target_col]


    cols_to_drop = [id_col, target_col, bias_col, 'age', 'SB', 'STach', 'CD']

    X = data.drop(columns=cols_to_drop, errors='ignore')

    return X, y

def train_model(X, y, config):
    model_type = config['experiment'].get('model_type', 'random_forest')
    seed = config['experiment']['random_seed']
    
    print(f"training: {model_type.upper()}...")
    
    if model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=seed)
    
    elif model_type == "logistic_regression":
        model = LogisticRegression(random_state=seed, max_iter=1500)
    
    else:
        print(f"error: model '{model_type}' not recognized!")
        exit()
        
    model.fit(X, y)
    return model



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a model to predict medical conditions from ECG embeddings.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to')

    args = parser.parse_args()

    print("using config file:", args.config)


    config = load_config(args.config)

    data = load_data(
            config['data']['embeddings_path'], 
            config['data']['metadata_path']
            ,config['data']['join_key']
        )
    
    bias_col = config['experiment']['bias_attribute']
    train_group = config['experiment']['train_group']
    test_group = config['experiment']['test_group']


    print(f"Filtering data for training group: {train_group} and testing group: {test_group}")

    df_group_A = data[data[bias_col] == train_group]
    df_group_B = data[data[bias_col] == test_group]

    X_A, y_A = prepare_features(df_group_A, config)
    X_train, X_test_same_group, y_train, y_test_same_group = train_test_split(
        X_A, y_A, test_size=0.2, random_state=42, stratify=y_A
    )

    X_test_other_group, y_test_other_group = prepare_features(df_group_B, config)

   

    print(f"training on {len(X_train)} exemples of group {train_group}...")
    model = train_model(X_train, y_train, config)

    pred_same = model.predict(X_test_same_group)
    acc_same = accuracy_score(y_test_same_group, pred_same)

    pred_other = model.predict(X_test_other_group)
    acc_other = accuracy_score(y_test_other_group, pred_other)

    gap = acc_same - acc_other

    
    

    print("\n" + "="*40)
    print(f" Results for bias attribute: {bias_col}")
    print("="*40)
    print(f" base accuracy  ({train_group} -> {train_group}):  {acc_same:.2%}")
    print(f" transfer accuracy ({train_group} -> {test_group}): {acc_other:.2%}")
    print("-" * 40)
    print(f" GAP (Bias): {gap:.2%} (Positive = Favors {train_group})")
    print("="*40)