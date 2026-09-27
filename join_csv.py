import pandas as pd

def unisci_csv_senza_duplicati(path_file1, path_file2, path_output):
    try:
        df1 = pd.read_csv(path_file1)
        df2 = pd.read_csv(path_file2)
        
        df_unito = pd.concat([df1, df2], ignore_index=True)
        
        righe_totali_prima = len(df_unito)
        
        df_pulito = df_unito.drop_duplicates()
        
        # Conteggio finale
        righe_totali_dopo = len(df_pulito)
        duplicati_trovati = righe_totali_prima - righe_totali_dopo
        
        print(f"--- Resoconto ---")
        print(f"Righe duplicate trovate e rimosse: {duplicati_trovati}")
        print(f"Righe totali nel file finale: {righe_totali_dopo}")
        
        # 4. Salva il risultato nel nuovo file CSV
        # index=False evita di aggiungere una colonna extra con i numeri di riga
        df_pulito.to_csv(path_output, index=False)
        print(f"File unito salvato con successo in: {path_output}")
        
    except Exception as e:
        print(f"Si è verificato un errore durante l'elaborazione: {e}")

# --- COME USARE IL PROGRAMMA ---
# Sostituisci i percorsi d'esempio con i percorsi reali dei tuoi file
path1 = "checkpoints/random_forest/chapman_ecgfm/AF_sex/debias_post_task7/removed_features_step_32.csv"
path2 = "checkpoints/random_forest/chapman_ecgfm/AF_sex/efficiency_ablation/removed_features_700.csv"
output = "RF_chapman_ecgfm_700_32.csv"

unisci_csv_senza_duplicati(path1, path2, output)