import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
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

if __name__ == "__main__":
    generate_perfect_plots()