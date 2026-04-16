import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ConfigLoader import ConfigLoader
from DataLoader import DataLoader
import os

def generate_perfect_plots(csv_path="results/experiments_log.csv"):
    if not os.path.exists(csv_path):
        print(f"Errore: Il file {csv_path} non esiste.")
        return

    df = pd.read_csv(csv_path)
    
    def extract_n(task_text):

        nums = re.findall(r'\d+', str(task_text))
        return int(nums[-1]) if len(nums) >= 2 else None

    # Filtro Task
    df_t4 = df[df['Task'].str.contains("TASK 4", na=False)].copy()
    df_t7 = df[df['Task'].str.contains("TASK 7", na=False)].copy()

    tasks = {"task4": df_t4, "task7": df_t7}
    sns.set_style("whitegrid")

    for task_id, data in tasks.items():
        if data.empty: continue
            
        data['N_Removed'] = data['Task'].apply(extract_n)
        # Rimozione eventuali righe dove non trovo il numero
        data = data.dropna(subset=['N_Removed'])
        
        for (ds, model), group in data.groupby(['Dataset', 'Model']):
            # Ordinamento dati (25, 50, 75...)
            group = group.sort_values('N_Removed')
            
            # Valori esatti per l'asse X
            exact_ticks = group['N_Removed'].unique()
            
            target_dir = os.path.join("results", f"grafici_{task_id}", model)
            os.makedirs(target_dir, exist_ok=True)
            
            plt.figure(figsize=(12, 6))
            
            # Plot Maschi
            plt.plot(group['N_Removed'], group['F1_Male_Med'], 
                     marker='o', markersize=8, linewidth=2.5, label="Maschi (F1)", color='#1f77b4')
            
            # Plot Femmine
            plt.plot(group['N_Removed'], group['F1_Female_Med'], 
                     marker='s', markersize=8, linestyle='--', linewidth=2.5, label="Femmine (F1)", color='#d62728')

            plt.xticks(exact_ticks, rotation=45)
            # Forzo il grafico a iniziare dal primo punto e finire all'ultimo
            plt.xlim(min(exact_ticks), max(exact_ticks)) 

            plt.title(f"{task_id.upper()}: {ds}\n({model})", fontsize=14, fontweight='bold')
            plt.xlabel("Numero di Feature Rimosse", fontsize=12)
            plt.ylabel("F1-Score (Mediana)", fontsize=12)
            
            # Y dinamica per vedere bene i cambiamenti
            all_f1 = pd.concat([group['F1_Male_Med'], group['F1_Female_Med']])
            plt.ylim(all_f1.min() - 0.03, min(1.0, all_f1.max() + 0.03)) 

            plt.legend(loc='best')
            plt.tight_layout()
            
            save_path = os.path.join(target_dir, f"{ds}.png")
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"Grafico corretto salvato: {save_path} (Punti X: {exact_ticks})")

def generate_pca_plots():

    config = ConfigLoader("config.yaml")
    data_loader = DataLoader(config)

    target_dir = os.path.join("results", "pca_plots","embeddings_non_filtrati")
    os.makedirs(target_dir, exist_ok=True)

    datasets = ["georgia_ecgfm", "chapman_ecgfm", "georgia_hubert_ecg_base", "chapman_hubert_ecg_base"]

    sns.set_context("notebook", font_scale=1.1)
    sns.set_style("ticks")

    for ds in datasets:
        print(f"Generazione PCA per {ds}...")
        embedding_path = None
        metadata_path = None
        
        for fm in config.data.foundation_models:
            for ds_info in fm.get('datasets', []):
                if ds_info['name'] == ds:
                    embedding_path = ds_info['embedding_path']
                    metadata_path = ds_info['metadata_path']
                    break
            if embedding_path: break
            
        if not embedding_path:
            print(f"Errore: Percorsi non trovati nel config per {ds}")
            continue

        config.data.current_dataset = ds
        config.data.embedding_path = embedding_path  
        config.data.metadata_path = metadata_path

        try:
            data = data_loader.load_data()
            balanced_data = data_loader.balance_model(data)
            X_scaled, y = data_loader.prepare_features(balanced_data)

            sex_values = balanced_data['sex'].values

            target_col = config.experiment.target_column[0]

            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            # percentuale varianza spiegata
            var_explained = pca.explained_variance_ratio_ * 100

            pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
            pca_df['Diagnosi'] = y
            pca_df['Sesso'] = sex_values

            diagnosis_mapping = {0: 'Sano', 1: 'AF'}
            pca_df['Diagnosi'] = pca_df['Diagnosi'].map(diagnosis_mapping)
            pca_df['Sesso'] = pca_df['Sesso'].map({0: 'Maschio', 1: 'Femmina'})

            pca_df['Gruppo'] = pca_df['Sesso'] + " - " + pca_df['Diagnosi']

            plt.figure(figsize=(12, 8))

            custom_palette = {
                "Maschio - Sano": "#6baed6",
                "Maschio - AF": "#08519c",   
                "Femmina - Sano": "#fb6a4a",  
                "Femmina - AF": "#a50f15"    
            }

            custom_markers = {
                "Maschio - Sano": "o",
                "Maschio - AF": "o",
                "Femmina - Sano": "s",
                "Femmina - AF": "s"
            }

            scatter = sns.scatterplot(
                data = pca_df,
                x = 'PC1',
                y = 'PC2',
                hue = 'Gruppo',
                style='Gruppo',
                palette=custom_palette, 
                markers=custom_markers,
                s = 80,
                alpha = 0.6,
                linewidth = 0.5,
            )

            plt.xlabel(f"PC1 - Varianza: {var_explained[0]:.1f}%", fontsize=12, fontweight='bold')
            plt.ylabel(f"PC2 - Varianza: {var_explained[1]:.1f}%", fontsize=12, fontweight='bold')

            plt.title(f"Proiezione PCA 2D | Dataset: {ds}\nTarget: {target_col}", fontsize=14, fontweight='bold')
            plt.legend(title='Gruppo Pazienti', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)

            plt.axhline(0, color='grey', linestyle='--', linewidth=1)
            plt.axvline(0, color='grey', linestyle='--', linewidth=1)

            sns.despine()
            plt.tight_layout()

            save_path = os.path.join(target_dir, f"{ds}.png")
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"Salvato grafico PCA 2D: {save_path}")
        
        except Exception as e:
            print(f"Errore su {ds}: {e}")
            plt.close()

def generate_task8_pca_plots():
    config = ConfigLoader("config.yaml")
    data_loader = DataLoader(config)

    model = config.experiment.model_type

    target_dir = os.path.join("results", "pca_plots", "embeddings_filtrati_task8", model)
    os.makedirs(target_dir, exist_ok=True)

    datasets_to_process = ["georgia_ecgfm", "chapman_ecgfm", "georgia_hubert_ecg_base", "chapman_hubert_ecg_base"]

    sns.set_context("notebook", font_scale=1.1)
    sns.set_style("ticks")

    for ds_name in datasets_to_process:
        print(f"\nGenerazione PCA Task 8 per {ds_name}...")
        
        embedding_path = None
        metadata_path = None
        for fm in config.data.foundation_models:
            for ds_info in fm.get('datasets', []):
                if ds_info['name'] == ds_name:
                    embedding_path = ds_info['embedding_path']
                    metadata_path = ds_info['metadata_path']
                    break
            if embedding_path: break

        if not embedding_path:
            continue

        config.data.current_dataset = ds_name
        config.data.embedding_path = embedding_path  
        config.data.metadata_path = metadata_path
        config.data.embeddings_path = embedding_path

        try:
            model_type = config.experiment.model_type
            if not hasattr(config.experiment, ds_name) or model_type not in getattr(config.experiment, ds_name):
                print(f" [!] Parametri best non trovati per {ds_name}. Salto.")
                continue
                
            model_params = getattr(config.experiment, ds_name)[model_type]
            best_t4 = model_params['best_t4']
            best_t7 = model_params['best_t7']
            
            target = config.experiment.target_column[0]
            bias = config.experiment.bias_attribute[0]
            
            # 3. Leggi i file CSV generati dalle run precedenti
            path_t4 = f"checkpoints/{model_type}/{ds_name}/{target}_{bias}/shap_ablation/features_to_remove_{best_t4}.csv"
            path_t7 = f"checkpoints/{model_type}/{ds_name}/{target}_{bias}/efficiency_ablation/removed_features_{best_t7}.csv"
            
            if not os.path.exists(path_t4) or not os.path.exists(path_t7):
                print(f" [!] File CSV delle run precedenti mancanti per {ds_name}. Assicurati di aver eseguito Task 4 e Task 7.")
                continue

            feat_t4_proxies = set(pd.read_csv(path_t4)['feature_to_remove'].tolist())
            feat_t7_removed = set(pd.read_csv(path_t7)['removed_feature'].tolist())
            
            all_features_to_remove = list(feat_t4_proxies.union(feat_t7_removed))
            print(f" -> Trovate {len(all_features_to_remove)} feature TOTALI da rimuovere (Bias + Inutili).")

            data = data_loader.load_data()
            balanced_data = data_loader.balance_model(data) 
            
            balanced_data_pruned = balanced_data.drop(columns=all_features_to_remove, errors='ignore')
            
            X_scaled, y = data_loader.prepare_features(balanced_data_pruned, train=True)
            sex_values = balanced_data_pruned['sex'].values

            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            var_explained = pca.explained_variance_ratio_ * 100

            pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
            pca_df['Diagnosi_Num'] = y
            pca_df['Sesso_Num'] = sex_values

            pca_df['Diagnosi'] = pca_df['Diagnosi_Num'].map({0: 'Sano', 1: 'AF'})
            pca_df['Sesso'] = pca_df['Sesso_Num'].map({0: 'Maschio', 1: 'Femmina'})
            pca_df['Gruppo'] = pca_df['Sesso'] + " - " + pca_df['Diagnosi']

            plt.figure(figsize=(12, 8))
            
            custom_palette = {
                "Maschio - Sano": "#6baed6",               
                "Maschio - AF": "#08519c",   
                "Femmina - Sano": "#fb6a4a",               
                "Femmina - AF": "#a50f15"    
            }
            
            custom_markers = {
                "Maschio - Sano": "o",
                "Maschio - AF": "o",
                "Femmina - Sano": "s",
                "Femmina - AF": "s"
            }

            sns.scatterplot(
                data=pca_df, x='PC1', y='PC2', 
                hue='Gruppo', style='Gruppo',    
                palette=custom_palette, markers=custom_markers,
                s=80, alpha=0.7, linewidth=0.5
            )

            plt.xlabel(f"PC1 - Varianza Spiegata: {var_explained[0]:.1f}%", fontsize=12, fontweight='bold')
            plt.ylabel(f"PC2 - Varianza Spiegata: {var_explained[1]:.1f}%", fontsize=12, fontweight='bold')

            num_feature_rimaste = X_scaled.shape[1]
            plt.title(f"PCA 2D POST TASK 8 (Embedding Filtrati)\nDataset: {ds_name} | Numero feature: {num_feature_rimaste}", fontsize=14, fontweight='bold')
            plt.legend(title='Gruppo Pazienti', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
            
            plt.axhline(0, color='grey', linestyle='--', linewidth=1)
            plt.axvline(0, color='grey', linestyle='--', linewidth=1)

            sns.despine()
            plt.tight_layout()

            save_path = os.path.join(target_dir, f"{ds_name}_task8.png")
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"Salvato grafico PCA Task 8: {save_path}")
        
        except Exception as e:
            print(f"Errore su {ds_name}: {e}")
            plt.close()


if __name__ == "__main__":
    print("Generazione grafici...")

    #generate_perfect_plots()

    #generate_pca_plots()

    generate_task8_pca_plots()
