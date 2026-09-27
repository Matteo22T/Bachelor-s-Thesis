import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ConfigLoader import ConfigLoader
from DataLoader import DataLoader
from datetime import datetime
import os
import numpy as np


def generate_perfect_plots(csv_path="results/experiments_log.csv"):
    if not os.path.exists(csv_path):
        print(f"Errore: Il file {csv_path} non esiste.")
        return

    df = pd.read_csv(csv_path)
    
    def extract_n(task_text):

        nums = re.findall(r'\d+', str(task_text))
        return int(nums[-1]) if len(nums) >= 2 else None

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
                     marker='o', markersize=8, linewidth=2.5, label="Male (F1)", color='#1f77b4')
            
            # Plot Femmine
            plt.plot(group['N_Removed'], group['F1_Female_Med'], 
                     marker='s', markersize=8, linestyle='--', linewidth=2.5, label="Female (F1)", color='#d62728')

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

def generate_debias_post_task7_plots():
    sns.set_context("notebook", font_scale=1.1)
    sns.set_style("whitegrid")

    base_dir = "results/debias_post_task7"  
    if not os.path.exists(base_dir):
        print("Nessun dato trovato per la Task debiasing post-Task 7.")
        return

    for model in os.listdir(base_dir):
        model_dir = os.path.join(base_dir, model)
        if not os.path.isdir(model_dir): continue

        for file in os.listdir(model_dir):
            if not file.endswith(".csv"): continue
            
            ds_name = file.replace(".csv", "")
            df = pd.read_csv(os.path.join(model_dir, file))

            plt.figure(figsize=(12, 7))

            plt.plot(df['N_Removed'], df['F1_Male_Med'], marker='o', markersize=5, linewidth=1.5, color='#1f77b4', label='Diagnosis AF - Male (F1)')
            
            plt.plot(df['N_Removed'], df['F1_Female_Med'], marker='s', markersize=5, linestyle='--', linewidth=1.5, color='#d62728', label='Diagnosis AF - Female (F1)')
            
            plt.plot(df['N_Removed'], df['F1_Sesso_Macro'], marker='^', markersize=5, linestyle=':', linewidth=2, color='#2ca02c', label='Diagnosis Sex (F1 Macro)')
            plt.axhline(y=0.5, color='grey', linestyle='--', alpha=0.8, label='Modello cieco (Random Guess 50%)')

            plt.title(f"De-biasing Analysis: Diagnosis Performance vs Recognition of Sex, Post-Task 7\nDataset: {ds_name} | Modello: {model}", fontsize=14, fontweight='bold')
            plt.xlabel("Number of Proxy Features Removed (Non-linear Scale: 0, 2, 4, 8...)", fontsize=12)
            plt.ylabel("F1-Score (Median)", fontsize=12)

            exact_ticks = df['N_Removed'].values
            plt.xticks(exact_ticks, rotation=0)
            plt.ylim(0.4, 1.05) 

            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()

            out_plot = os.path.join("results/grafici_debias_post_task7", model)
            os.makedirs(out_plot, exist_ok=True)
            save_path = os.path.join(out_plot, f"{ds_name}.png")
            plt.savefig(save_path, dpi=300)
            plt.close()

def generate_pca_plots():
    config = ConfigLoader("config.yaml")
    data_loader = DataLoader(config)

    model_type = config.experiment.model_type
    
    base_target_dir = os.path.join("results", "pca_plots", "embeddings_filtrati")
    img_target_dir = os.path.join(base_target_dir, model_type)
    os.makedirs(img_target_dir, exist_ok=True)
    
    master_csv_path = os.path.join(base_target_dir, "pca_variance_log.csv")

    datasets = ["georgia_ecgfm", "chapman_ecgfm", "georgia_hubert_ecg_base", "chapman_hubert_ecg_base"]
    n_comp = getattr(config.experiment, 'pca_dimensions', 2)

    sns.set_context("notebook", font_scale=1.1)
    sns.set_style("ticks")
    
    variance_log_data = []

    for ds in datasets:
        print(f"\nGenerazione PCA ({n_comp}D) per {ds}...")
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
            continue

        config.data.current_dataset = ds
        config.data.embeddings_path = embedding_path  
        config.data.metadata_path = metadata_path

        try:
            if not hasattr(config.experiment, ds) or model_type not in getattr(config.experiment, ds):
                print(f" [!] Parametri best non trovati per {ds}. Salto.")
                continue
                
            model_params = getattr(config.experiment, ds)[model_type]
            best_t7 = model_params.get('best_t7')
            best_t4_post = model_params.get('best_t4_post_t7')
            
            if best_t7 is None or best_t4_post is None:
                print(f" [!] Parametri 'best_t7' o 'best_t4_post_t7' mancanti nel config per {ds}. Salto.")
                continue
            
            target = config.experiment.target_column[0]
            bias = config.experiment.bias_attribute[0]
            
            path_t7 = f"checkpoints/{model_type}/{ds}/{target}_{bias}/efficiency_ablation/removed_features_{best_t7}.csv"
            path_t4_post = f"checkpoints/{model_type}/{ds}/{target}_{bias}/debias_post_task7/removed_features_step_{best_t4_post}.csv"
            
            feat_t7_removed = set()
            feat_t4_post_removed = set()
            
            if os.path.exists(path_t7):
                feat_t7_removed = set(pd.read_csv(path_t7)['removed_feature'].tolist())
            else:
                print(f" [!] Manca file Task 7: {path_t7}")
                continue
                
            if best_t4_post > 0:
                if os.path.exists(path_t4_post):
                    feat_t4_post_removed = set(pd.read_csv(path_t4_post)['removed_feature'].tolist())
                else:
                    print(f" [!] Manca file Debias post T7: {path_t4_post}")
                    continue
            
            all_features_to_remove = list(feat_t7_removed.union(feat_t4_post_removed))
            print(f" -> Feature da scartare: {len(all_features_to_remove)} (T7: {len(feat_t7_removed)} | Debias: {len(feat_t4_post_removed)})")

            data = data_loader.load_data()
            balanced_data = data_loader.balance_model(data, max_samples=200)
            
            balanced_data_pruned = balanced_data.drop(columns=all_features_to_remove, errors='ignore')
            
            X_scaled, y = data_loader.prepare_features(balanced_data_pruned, train=True)
            sex_values = balanced_data_pruned['sex'].values
            num_feature_rimaste = X_scaled.shape[1]

            num_samples = X_scaled.shape[0]
            print(f" -> Dati passati alla PCA: {num_samples} pazienti (su {len(data)} originali)")

            pca = PCA(n_components=n_comp)
            X_pca = pca.fit_transform(X_scaled)
            
            var_explained = pca.explained_variance_ratio_ * 100
            total_variance = var_explained.sum()

            variance_log_data.append({
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Dataset': ds,
                'Model': model_type,
                'PCA_Dimensions': n_comp,
                'T7_Removed': best_t7,
                'T4_Post_Removed': best_t4_post,
                'Features_Kept': num_feature_rimaste,
                'Samples_Used': num_samples,  # <--- NUOVA COLONNA
                'Total_Variance_%': round(total_variance, 4)
            })

            if n_comp == 2:
                pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
                pca_df['Diagnosi'] = y
                pca_df['Sesso'] = sex_values

                pca_df['Diagnosi'] = pca_df['Diagnosi'].map({0: 'Healthy', 1: 'AF'})
                pca_df['Sesso'] = pca_df['Sesso'].map({0: 'Male', 1: 'Female'})
                pca_df['Gruppo'] = pca_df['Sesso'] + " - " + pca_df['Diagnosi']

                plt.figure(figsize=(12, 8))

                custom_palette = {
                    "Male - Healthy": "#6baed6",
                    "Male - AF": "#08519c",   
                    "Female - Healthy": "#fb6a4a",  
                    "Female - AF": "#a50f15"    
                }
                custom_markers = {
                    "Male - Healthy": "o", "Male - AF": "o",
                    "Female - Healthy": "s", "Female - AF": "s"
                }

                sns.scatterplot(
                    data=pca_df, x='PC1', y='PC2',
                    hue='Gruppo', style='Gruppo',
                    palette=custom_palette, markers=custom_markers,
                    s=80, alpha=0.7, linewidth=0.5
                )
                plt.xlabel(f"PC1 - Variance explained: {var_explained[0]:.1f}%", fontsize=16, fontweight='bold')
                plt.ylabel(f"PC2 - Variance explained: {var_explained[1]:.1f}%", fontsize=16, fontweight='bold')

                plt.title(f"PCA 2D (Filtered Embedding)\nDataset: {ds} | Feature: {num_feature_rimaste} | Samples: {num_samples}", fontsize=18, fontweight='bold')

                plt.tick_params(axis='both', which='major', labelsize=14)

                plt.legend(title='Patient Group', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)

                plt.axhline(0, color='grey', linestyle='--', linewidth=1)
                plt.axvline(0, color='grey', linestyle='--', linewidth=1)

                sns.despine()
                plt.tight_layout()

                file_name = f"{ds}_t7_{best_t7}_t4post_{best_t4_post}.png"
                save_path = os.path.join(img_target_dir, file_name)
                plt.savefig(save_path, dpi=300)
                plt.close()
                print(f" -> Salvato grafico PCA 2D: {save_path}")

        except Exception as e:
            print(f"Errore su {ds}: {e}")
            plt.close()
            
    if variance_log_data:
        df_new_log = pd.DataFrame(variance_log_data)
        file_exists = os.path.exists(master_csv_path)
        df_new_log.to_csv(master_csv_path, mode='a', index=False, header=not file_exists)

def generate_task8_pca_plots():
    config = ConfigLoader("config.yaml")
    data_loader = DataLoader(config)

    model = config.experiment.model_type

    target_dir = os.path.join("results", "pca_plots", "embeddings_filtrati_task8", model)
    os.makedirs(target_dir, exist_ok=True)

    datasets_to_process = ["georgia_ecgfm", "chapman_ecgfm", "georgia_hubert_ecg_base", "chapman_hubert_ecg_base"]

    sns.set_context("notebook", font_scale=1.4)
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

            pca_df['Diagnosi'] = pca_df['Diagnosi_Num'].map({0: 'Healthy', 1: 'AF'})
            pca_df['Sesso'] = pca_df['Sesso_Num'].map({0: 'Male', 1: 'Female'})
            pca_df['Gruppo'] = pca_df['Sesso'] + " - " + pca_df['Diagnosi']

            plt.figure(figsize=(12, 8))
            
            custom_palette = {
                "Male - Healthy": "#6baed6",               
                "Male - AF": "#08519c",   
                "Female - Healthy": "#fb6a4a",               
                "Female - AF": "#a50f15"    
            }
            
            custom_markers = {
                "Male - Healthy": "o",
                "Male - AF": "o",
                "Female - Healthy": "s",
                "Female - AF": "s"
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


def generate_cross_dataset_plots(log_csv_path="results/experiments_log.csv"):
    try:
        df = pd.read_csv(log_csv_path)
    except FileNotFoundError:
        print(f"Errore: Il file {log_csv_path} non è stato trovato.")
        return

    plt.rcParams.update({
        'font.size': 14, 
        'axes.labelsize': 16, 
        'legend.fontsize': 12, 
        'xtick.labelsize': 13, 
        'ytick.labelsize': 14
    })
    
    models = ['random_forest', 'logistic_regression']
    
    dataset_groups = {
        'ecgfm': ('georgia_ecgfm', 'chapman_ecgfm'),
        'hubert': ('georgia_hubert_ecg_base', 'chapman_hubert_ecg_base')
    }

    best_t4_dict = {
        'georgia_ecgfm': {'logistic_regression': 32, 'random_forest': 32},
        'chapman_ecgfm': {'logistic_regression': 32, 'random_forest': 32},
        'georgia_hubert_ecg_base': {'logistic_regression': 16, 'random_forest': 16},
        'chapman_hubert_ecg_base': {'logistic_regression': 16, 'random_forest': 16}
    }

    def get_f1_macro(dataset_val, task_substr, model_val):
        row = df[(df['Dataset'] == dataset_val) & 
                 (df['Model'] == model_val) & 
                 (df['Task'].str.contains(task_substr, case=False, na=False)) &
                 (~df['Task'].str.contains('RANDOM', case=False, na=False))]
        if not row.empty:
            f1_m = row['F1_Male_Med'].values[-1]
            f1_f = row['F1_Female_Med'].values[-1]
            return (f1_m + f1_f) / 2
        return 0.0

    colors = ['#B0BEC5', '#546E7A', '#EF5350', '#FFA726', '#42A5F5']
    x_labels = [
        'In-Domain\nBase', 
        'In-Domain\nDebiased', 
        'Cross-Domain\n(Base)', 
        'Cross-Domain\n(Filtered)', 
        'Cross-Domain\n(Union)'
    ]

    for fm_name, (ds_geo, ds_cha) in dataset_groups.items():
        for model in models:
            
            # --- RACCOLTA DATI: TRAIN GEORGIA -> TEST CHAPMAN ---
            best_t4_geo = best_t4_dict[ds_geo][model]
            str_cross_geo_to_cha = f"Train:{ds_geo} -> Test:{ds_cha}"
            
            y_georgia = [
                get_f1_macro(ds_geo, 'CASO BASE', model),
                get_f1_macro(ds_geo, f'Debiasing \(Pareto: -{best_t4_geo} proxy\)', model),
                get_f1_macro(str_cross_geo_to_cha, 'Parte 1', model),
                get_f1_macro(str_cross_geo_to_cha, 'Parte 2', model),
                get_f1_macro(str_cross_geo_to_cha, 'Parte 3', model)
            ]

            # --- RACCOLTA DATI: TRAIN CHAPMAN -> TEST GEORGIA ---
            best_t4_cha = best_t4_dict[ds_cha][model]
            str_cross_cha_to_geo = f"Train:{ds_cha} -> Test:{ds_geo}"
            
            y_chapman = [
                get_f1_macro(ds_cha, 'CASO BASE', model),
                get_f1_macro(ds_cha, f'Debiasing \(Pareto: -{best_t4_cha} proxy\)', model),
                get_f1_macro(str_cross_cha_to_geo, 'Parte 1', model),
                get_f1_macro(str_cross_cha_to_geo, 'Parte 2', model),
                get_f1_macro(str_cross_cha_to_geo, 'Parte 3', model)
            ]

            x = np.arange(5)

            # --- GRAFICO 1: TRAIN GEORGIA ---
            fig, ax = plt.subplots(figsize=(10, 6))
            bars_g = ax.bar(x, y_georgia, color=colors, edgecolor='black', zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, fontweight='bold')
            ax.set_ylabel('Macro F1 Score', fontweight='bold')
            ax.set_title(f'Train: Georgia $\\rightarrow$ Test: Chapman\n({fm_name.upper()} - {model})', fontweight='bold', pad=15)
            ax.set_ylim(0, 1.05)
            
            for bar, val in zip(bars_g, y_georgia):
                if val >= 0:
                    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.2f}", ha='center', va='bottom', fontsize=11, fontweight='bold')

            plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
            plt.tight_layout()
            plt.savefig(f"paper_cross_{fm_name}_{model}_train_georgia.png", format='png', dpi=300, bbox_inches='tight')
            plt.close()

            # --- GRAFICO 2: TRAIN CHAPMAN ---
            fig, ax = plt.subplots(figsize=(10, 6))
            bars_c = ax.bar(x, y_chapman, color=colors, edgecolor='black', zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, fontweight='bold')
            ax.set_ylabel('Macro F1 Score', fontweight='bold')
            ax.set_title(f'Train: Chapman $\\rightarrow$ Test: Georgia\n({fm_name.upper()} - {model})', fontweight='bold', pad=15)
            ax.set_ylim(0, 1.05)
            
            for bar, val in zip(bars_c, y_chapman):
                if val >= 0:
                    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.2f}", ha='center', va='bottom', fontsize=11, fontweight='bold')

            plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
            plt.tight_layout()
            plt.savefig(f"paper_cross_{fm_name}_{model}_train_chapman.png", format='png', dpi=300, bbox_inches='tight')
            plt.close()

    print("Generazione completata: grafici singoli salvati con successo.")

if __name__ == "__main__":
    print("Generazione grafici...")

    #generate_perfect_plots()

    #generate_pca_plots()

    #generate_task8_pca_plots()

    #generate_debias_post_task7_plots()

    generate_cross_dataset_plots()
