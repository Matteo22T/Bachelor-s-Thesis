import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_paper_task7_plots(log_csv_path="experiments_log.csv"):
    try:
        df = pd.read_csv(log_csv_path)
    except FileNotFoundError:
        print(f"Errore: Il file {log_csv_path} non è stato trovato.")
        return

    # Impostazioni font
    plt.rcParams.update({
        'font.size': 20, 
        'axes.labelsize': 22, 
        'legend.fontsize': 14, 
        'xtick.labelsize': 20, 
        'ytick.labelsize': 20
    })
    
    models = ['random_forest', 'logistic_regression']
    
    # Suddivisione logica per Foundation Model
    dataset_groups = {
        'ecgfm': ['georgia_ecgfm', 'chapman_ecgfm'],
        'hubert': ['georgia_hubert_ecg_base', 'chapman_hubert_ecg_base']
    }
    
    # Colori coerenti: Georgia (Rosso), Chapman (Blu)
    colors = {
        'georgia_ecgfm_M': '#8B0000', 'georgia_ecgfm_F': '#FF6666',
        'chapman_ecgfm_M': '#00008B', 'chapman_ecgfm_F': '#66B2FF',
        'georgia_hubert_ecg_base_M': '#8B0000', 'georgia_hubert_ecg_base_F': '#FF6666',
        'chapman_hubert_ecg_base_M': '#00008B', 'chapman_hubert_ecg_base_F': '#66B2FF'
    }

    steps_t7 = [150, 300, 400, 512, 600, 650, 700, 758]
    x_labels = ['0'] + [str(s) for s in steps_t7] + ['768']
    x_positions = np.arange(len(x_labels))

    # Doppio ciclo: Prima il Foundation Model, poi il Classificatore
    for fm_name, datasets in dataset_groups.items():
        for model in models:
            fig, ax = plt.subplots(figsize=(12, 9))
            
            for ds in datasets:
                base = df[(df['Dataset'] == ds) & (df['Model'] == model) & (df['Task'].str.contains('CASO BASE', case=False, na=False))]
                f1_m_base = base['F1_Male_Med'].values[0] if not base.empty else np.nan
                f1_f_base = base['F1_Female_Med'].values[0] if not base.empty else np.nan
                
                # Inizializziamo le 4 curve
                y_male = [f1_m_base]
                y_female = [f1_f_base]
                y_male_rnd = [f1_m_base]
                y_female_rnd = [f1_f_base]
                
                for step in steps_t7:
                    # Estrazione dati REALI (SHAP)
                    mask_real = (df['Dataset'] == ds) & (df['Model'] == model) & (df['Task'].str.contains(f'-{step} feature', na=False)) & (~df['Task'].str.contains('RANDOM', case=False, na=False))
                    row = df[mask_real]                
                    y_male.append(row['F1_Male_Med'].values[0] if not row.empty else np.nan)
                    y_female.append(row['F1_Female_Med'].values[0] if not row.empty else np.nan)

                    # Estrazione dati RANDOM
                    mask_rnd = (df['Dataset'] == ds) & (df['Model'] == model) & (df['Task'].str.contains(f'-{step} feature', na=False)) & (df['Task'].str.contains('RANDOM', case=False, na=False))
                    row_rnd = df[mask_rnd]
                    y_male_rnd.append(row_rnd['F1_Male_Med'].values[0] if not row_rnd.empty else np.nan)
                    y_female_rnd.append(row_rnd['F1_Female_Med'].values[0] if not row_rnd.empty else np.nan)
                
                y_male.extend([np.nan])
                y_female.extend([np.nan])
                y_male_rnd.extend([np.nan])
                y_female_rnd.extend([np.nan])
                
                ds_label = "Georgia" if "georgia" in ds else "Chapman-Ningbo"
                
                # --- PLOT RANDOM (Dietro, tratteggiato, con le "x") ---
                ax.plot(x_positions, y_male_rnd, label=f'RND AF-Male ({ds_label})', 
                        color=colors[f'{ds}_M'], linewidth=2, linestyle='--', marker='x', markersize=8, alpha=0.7)
                
                ax.plot(x_positions, y_female_rnd, label=f'RND AF-Female ({ds_label})', 
                        color=colors[f'{ds}_F'], linewidth=2, linestyle='--', marker='x', markersize=8, alpha=0.7)

                # --- PLOT REALE (Davanti, spesso, marcatori pieni) ---
                ax.plot(x_positions, y_male, label=f'AF - Male ({ds_label})', 
                        color=colors[f'{ds}_M'], linewidth=4, marker='o', markersize=10)
                
                ax.plot(x_positions, y_female, label=f'AF - Female ({ds_label})', 
                        color=colors[f'{ds}_F'], linewidth=4, marker='s', markersize=10)

            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels)
            ax.set_xlim(0, len(x_positions) - 1) 

            ax.set_xlabel("Removed diagnostic features (n)", fontweight='bold')
            ax.set_ylabel("F1 score", fontweight='bold')
            
            ax.legend(loc='lower left', frameon=True, fontsize=15, ncol=2)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
        

            # Nome file dinamico per salvare i 4 grafici distintamente
            file_name = f"paper_task7_{fm_name}_{model}_combined.png"
            plt.savefig(file_name, format='png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Salvato plot Task 7 per {fm_name.upper()} - {model} (PNG, 300 DPI)")
            
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_paper_debias_plots(georgia_csv_path, chapman_csv_path, georgia_rnd_path, chapman_rnd_path, model_name="random_forest", fm_name="ecgfm", max_features_label="68"):
    try:
        df_geo = pd.read_csv(georgia_csv_path)
        df_cha = pd.read_csv(chapman_csv_path)
        df_geo_rnd = pd.read_csv(georgia_rnd_path)
        df_cha_rnd = pd.read_csv(chapman_rnd_path)
    except FileNotFoundError as e:
        print(f"Errore nel caricamento dei CSV per il Debias: {e}")
        return
    
    plt.rcParams.update({
        'font.size': 20, 
        'axes.labelsize': 22, 
        'legend.fontsize': 12,
        'xtick.labelsize': 20, 
        'ytick.labelsize': 20
    })
    
    colors = {
        'geo_M': '#8B0000', 'geo_F': '#FF6666', 'geo_S': '#FF0000',
        'cha_M': '#00008B', 'cha_F': '#66B2FF', 'cha_S': '#0000FF'
    }

    fig, ax = plt.subplots(figsize=(12, 9)) 
    removed_steps = df_geo['N_Removed'].tolist()
    
    # Tick finale dinamico (sostituisce il '68' fisso)
    x_labels = [str(n) for n in removed_steps] + [str(max_features_label)]
    x_pos = np.arange(len(x_labels))
    
    y_geo_m = df_geo['F1_Male_Med'].tolist() + [np.nan]
    y_geo_f = df_geo['F1_Female_Med'].tolist() + [np.nan]
    y_geo_s = df_geo['F1_Sesso_Macro'].tolist() + [np.nan]

    y_cha_m = df_cha['F1_Male_Med'].tolist() + [np.nan]
    y_cha_f = df_cha['F1_Female_Med'].tolist() + [np.nan]
    y_cha_s = df_cha['F1_Sesso_Macro'].tolist() + [np.nan]
    
    y_geo_m_rnd = df_geo_rnd['F1_Male_Med'].tolist() + [np.nan]
    y_geo_f_rnd = df_geo_rnd['F1_Female_Med'].tolist() + [np.nan]
    y_geo_s_rnd = df_geo_rnd['F1_Sesso_Macro'].tolist() + [np.nan]

    y_cha_m_rnd = df_cha_rnd['F1_Male_Med'].tolist() + [np.nan]
    y_cha_f_rnd = df_cha_rnd['F1_Female_Med'].tolist() + [np.nan]
    y_cha_s_rnd = df_cha_rnd['F1_Sesso_Macro'].tolist() + [np.nan]

    # --- PLOT RANDOM (Dietro, Linea Punteggiata) ---
    # Georgia Random
    ax.plot(x_pos, y_geo_m_rnd, label='RND AF-Male (Georgia)', color=colors['geo_M'], linewidth=2, linestyle=':', marker='x', markersize=8, alpha=0.7)
    ax.plot(x_pos, y_geo_f_rnd, label='RND AF-Female (Georgia)', color=colors['geo_F'], linewidth=2, linestyle=':', marker='x', markersize=8, alpha=0.7)
    ax.plot(x_pos, y_geo_s_rnd, label='RND Sex (Georgia)', color=colors['geo_S'], linewidth=2, linestyle=':', marker='*', markersize=8, alpha=0.7)

    # Chapman Random
    ax.plot(x_pos, y_cha_m_rnd, label='RND AF-Male (Chapman-Ningbo)', color=colors['cha_M'], linewidth=2, linestyle=':', marker='x', markersize=8, alpha=0.7)
    ax.plot(x_pos, y_cha_f_rnd, label='RND AF-Female (Chapman-Ningbo)', color=colors['cha_F'], linewidth=2, linestyle=':', marker='x', markersize=8, alpha=0.7)
    ax.plot(x_pos, y_cha_s_rnd, label='RND Sex (Chapman-Ningbo)', color=colors['cha_S'], linewidth=2, linestyle=':', marker='*', markersize=8, alpha=0.7)


    # --- PLOT REALI (Davanti, Spesse) ---
    # Georgia Reale
    ax.plot(x_pos, y_geo_m, label='AF - Male (Georgia)', color=colors['geo_M'], linewidth=4, marker='o', markersize=10)
    ax.plot(x_pos, y_geo_f, label='AF - Female (Georgia)', color=colors['geo_F'], linewidth=4, marker='s', markersize=10)
    ax.plot(x_pos, y_geo_s, label='Sex (Georgia)', color=colors['geo_S'], linewidth=4, linestyle='--', marker='^', markersize=10)

    # Chapman Reale
    ax.plot(x_pos, y_cha_m, label='AF - Male (Chapman-Ningbo)', color=colors['cha_M'], linewidth=4, marker='o', markersize=10)
    ax.plot(x_pos, y_cha_f, label='AF - Female (Chapman-Ningbo)', color=colors['cha_F'], linewidth=4, marker='s', markersize=10)
    ax.plot(x_pos, y_cha_s, label='Sex (Chapman-Ningbo)', color=colors['cha_S'], linewidth=4, linestyle='--', marker='^', markersize=10)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlim(0, len(x_pos) - 1)

    ax.set_xlabel("Removed proxy features (m)", fontweight='bold')
    ax.set_ylabel("F1 score", fontweight='bold')
    
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    fig.subplots_adjust(top=0.78) 
    
    # Nome salvataggio dinamico
    save_filename = f"paper_debias_{fm_name}_{model_name}_combined.png"
    plt.savefig(save_filename, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Salvato plot Debias Post-Task 7 per {fm_name.upper()} - {model_name} (PNG, 300 DPI)")


if __name__ == "__main__":
    print("Generazione dei grafici per il Paper in corso...")
    
    #try:
        #generate_paper_task7_plots("results/experiments_log.csv")
    #except Exception as e:
    #    print(f"Errore nella generazione Task 7: {e}")
    
    # 1. ECG-FM - Random Forest
    generate_paper_debias_plots(
        georgia_csv_path="results/debias_post_task7/random_forest/georgia_ecgfm.csv", 
        chapman_csv_path="results/debias_post_task7/random_forest/chapman_ecgfm.csv",
        georgia_rnd_path="results/debias_post_task7/random_forest/georgia_ecgfm_random.csv", 
        chapman_rnd_path="results/debias_post_task7/random_forest/chapman_ecgfm_random.csv",
        model_name="random_forest",
        fm_name="ecgfm",
        max_features_label="68"
    )

    # 2. ECG-FM - Logistic Regression
    generate_paper_debias_plots(
        georgia_csv_path="results/debias_post_task7/logistic_regression/georgia_ecgfm.csv", 
        chapman_csv_path="results/debias_post_task7/logistic_regression/chapman_ecgfm.csv", 
        georgia_rnd_path="results/debias_post_task7/logistic_regression/georgia_ecgfm_random.csv", 
        chapman_rnd_path="results/debias_post_task7/logistic_regression/chapman_ecgfm_random.csv", 
        model_name="logistic_regression",
        fm_name="ecgfm",
        max_features_label="68"
    )

    # 3. HUBERT - Random Forest
    generate_paper_debias_plots(
        georgia_csv_path="results/debias_post_task7/random_forest/georgia_hubert_ecg_base.csv", 
        chapman_csv_path="results/debias_post_task7/random_forest/chapman_hubert_ecg_base.csv",
        georgia_rnd_path="results/debias_post_task7/random_forest/georgia_hubert_ecg_base_random.csv", 
        chapman_rnd_path="results/debias_post_task7/random_forest/chapman_hubert_ecg_base_random.csv",
        model_name="random_forest",
        fm_name="hubert",
        max_features_label="Max" # O il numero esatto se preferisci
    )

    # 4. HUBERT - Logistic Regression
    generate_paper_debias_plots(
        georgia_csv_path="results/debias_post_task7/logistic_regression/georgia_hubert_ecg_base.csv", 
        chapman_csv_path="results/debias_post_task7/logistic_regression/chapman_hubert_ecg_base.csv", 
        georgia_rnd_path="results/debias_post_task7/logistic_regression/georgia_hubert_ecg_base_random.csv", 
        chapman_rnd_path="results/debias_post_task7/logistic_regression/chapman_hubert_ecg_base_random.csv", 
        model_name="logistic_regression",
        fm_name="hubert",
        max_features_label="Max" # O il numero esatto se preferisci
    )
    
    print("Fatto! Controlla i file PNG generati.")