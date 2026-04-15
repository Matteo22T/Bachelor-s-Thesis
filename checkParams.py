import joblib
import os

def check_and_save_params(file_path, log_file):
    if not os.path.exists(file_path):
        messaggio_errore = f"[!] Errore: Il file non esiste -> {file_path}\n"
        print(messaggio_errore, end="")
        log_file.write(messaggio_errore)
        return

    try:
        miglior_modello = joblib.load(file_path)
        parametri = miglior_modello.get_params()
        
        nome_modello = type(miglior_modello).__name__

        testo_output = f"\nAnalisi del file: {file_path}\n"
        testo_output += f"Modello: {nome_modello}\n"

        if nome_modello == "RandomForestClassifier":
            testo_output += f" - n_estimators:      {parametri.get('n_estimators')}\n"
            testo_output += f" - max_depth:         {parametri.get('max_depth')}\n"
            testo_output += f" - min_samples_split: {parametri.get('min_samples_split')}\n"
        
        elif nome_modello == "LogisticRegression":
            testo_output += f" - C:        {parametri.get('C')}\n"
            testo_output += f" - solver:   {parametri.get('solver')}\n"
            testo_output += f" - penalty:  {parametri.get('penalty')}\n"
            testo_output += f" - max_iter: {parametri.get('max_iter')}\n"
            
        else:
            testo_output += f" - [!] Parametri completi: {parametri}\n"

        testo_output += f"{'-' * 80}\n"

        # Stampa a schermo e scrivi sul file
        print(testo_output, end="")
        log_file.write(testo_output)

    except Exception as e:
        errore = f"[!] Errore durante la lettura di {file_path}: {e}\n"
        print(errore, end="")
        log_file.write(errore)


if __name__ == "__main__":

    cartella_output = "results"
    nome_file = "report_generale_best_params.txt"
    
    if not os.path.exists(cartella_output):
        os.makedirs(cartella_output)
        print(f"[INFO] Creata la cartella: {cartella_output}")
    
    folds_da_controllare = [
        
        # --- RANDOM FOREST ---
        "checkpoints/random_forest/georgia_ecgfm/AF_sex/base_case/fold_0.joblib",
        "checkpoints/random_forest/georgia_ecgfm/AF_sex/base_case/fold_1.joblib",
        "checkpoints/random_forest/georgia_ecgfm/AF_sex/base_case/fold_2.joblib",
        
        "checkpoints/random_forest/chapman_ecgfm/AF_sex/base_case/fold_0.joblib",
        "checkpoints/random_forest/chapman_ecgfm/AF_sex/base_case/fold_1.joblib",
        "checkpoints/random_forest/chapman_ecgfm/AF_sex/base_case/fold_2.joblib",

        "checkpoints/random_forest/georgia_hubert_ecg_base/AF_sex/base_case/fold_0.joblib",
        "checkpoints/random_forest/georgia_hubert_ecg_base/AF_sex/base_case/fold_1.joblib",
        "checkpoints/random_forest/georgia_hubert_ecg_base/AF_sex/base_case/fold_2.joblib",

        "checkpoints/random_forest/chapman_hubert_ecg_base/AF_sex/base_case/fold_0.joblib",
        "checkpoints/random_forest/chapman_hubert_ecg_base/AF_sex/base_case/fold_1.joblib",
        "checkpoints/random_forest/chapman_hubert_ecg_base/AF_sex/base_case/fold_2.joblib",

        # --- LOGISTIC REGRESSION ---
        "checkpoints/logistic_regression/georgia_ecgfm/AF_sex/base_case/fold_0.joblib",
        "checkpoints/logistic_regression/georgia_ecgfm/AF_sex/base_case/fold_1.joblib",
        "checkpoints/logistic_regression/georgia_ecgfm/AF_sex/base_case/fold_2.joblib",

        "checkpoints/logistic_regression/chapman_ecgfm/AF_sex/base_case/fold_0.joblib",
        "checkpoints/logistic_regression/chapman_ecgfm/AF_sex/base_case/fold_1.joblib",
        "checkpoints/logistic_regression/chapman_ecgfm/AF_sex/base_case/fold_2.joblib",

        "checkpoints/logistic_regression/georgia_hubert_ecg_base/AF_sex/base_case/fold_0.joblib",
        "checkpoints/logistic_regression/georgia_hubert_ecg_base/AF_sex/base_case/fold_1.joblib",
        "checkpoints/logistic_regression/georgia_hubert_ecg_base/AF_sex/base_case/fold_2.joblib",

        "checkpoints/logistic_regression/chapman_hubert_ecg_base/AF_sex/base_case/fold_0.joblib",
        "checkpoints/logistic_regression/chapman_hubert_ecg_base/AF_sex/base_case/fold_1.joblib",
        "checkpoints/logistic_regression/chapman_hubert_ecg_base/AF_sex/base_case/fold_2.joblib"
    ]
    
    
    print(f"Avvio estrazione parametri. I risultati verranno salvati in: {os.path.join(cartella_output, nome_file)}")
    
    with open(os.path.join(cartella_output, nome_file), "w") as f:
        f.write("=== REPORT GENERALE BEST PARAMS (RF & LR) ===\n")
        
        for fold in folds_da_controllare:
            check_and_save_params(fold, f)
            
    print(f"\n[INFO] Finito! Controlla il file {os.path.join(cartella_output, nome_file)} per i risultati.")