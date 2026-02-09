import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import warnings
warnings.filterwarnings('ignore')
from data.preprocessing import load_and_preprocess_data
from data.splits import run_loocv
from evaluation.metrics import compute_aggregate_metrics
from evaluation.visualization import plot_all_results
from utils.io_utils import save_results

INPUT_CSV = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\OSIC_metrics_validation\improved_prediction\quality_balanced_dataset.csv"
)

OUTPUT_DIR = Path(
    r"X:\Francesca Saglimbeni\tesi\vesselsegmentation"
    r"\validation_pipeline"
    r"\validation_test_models\FVC_prediction_results"
)

# Feature esatte usate in improved_fvc_prediction.py
FEATURES = [
    'mean_peripheral_branch_volume_mm3',
    'peripheral_branch_density',
    'mean_peripheral_diameter_mm',
    'central_to_peripheral_diameter_ratio',
    'mean_lung_density_HU',
    'histogram_entropy',
]

TARGET = 'FVC_percent_week52'

# Hyperparameters MLP
HIDDEN_1 = 16          # primo hidden layer
HIDDEN_2 = 8           # secondo hidden layer (bottleneck)
DROPOUT = 0.2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4    # L2 regularization nell'ottimizzatore
EPOCHS_MAX = 500
PATIENCE = 100         # early stopping aumentato per piccoli dataset

# Split interno
VAL_FRACTION = 0.20    # ~5 pazienti per val
N_INNER_SPLITS = 10    # ripetizioni dello split train/val per stabilità

# Seed
SEED = 42

# Device (GPU se disponibile)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Stili grafici
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


def main():
    """Funzione principale"""
    print("\n" + "="*70)
    print("  MLP MULTI-FEATURE FVC% PREDICTION")
    print("  Target: FVC_percent_week52 (prognosi a 1 anno)")
    print(f"  Device: {DEVICE}")
    print("="*70)
    
    # Verifica se GPU è disponibile
    if torch.cuda.is_available():
        print(f"  GPU disponibile: {torch.cuda.get_device_name(0)}")
        print(f"  Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("  ⚠ GPU non disponibile, usando CPU")

    # Crea directory di output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Carica e preprocessa i dati
        print(f"\n{'─'*70}")
        print("  FASE 1: Caricamento e preprocessing dati")
        print(f"{'─'*70}")
        
        df_clean = load_and_preprocess_data(
            input_path=INPUT_CSV,
            features=FEATURES,
            target=TARGET
        )
        
        print(f"  Dataset finale: {len(df_clean)} pazienti, {len(FEATURES)} feature")
        print(f"  Target range: [{df_clean[TARGET].min():.1f}, {df_clean[TARGET].max():.1f}]")
        
        # 2. Esegui LOOCV
        print(f"\n{'─'*70}")
        print(f"  FASE 2: LOOCV ({len(df_clean)} fold, {N_INNER_SPLITS} inner splits)")
        print(f"{'─'*70}")
        
        results_df, all_importances, fold_curves = run_loocv(
            df=df_clean,
            features=FEATURES,
            target=TARGET,
            device=DEVICE,
            config={
                'hidden1': HIDDEN_1,
                'hidden2': HIDDEN_2,
                'dropout': DROPOUT,
                'learning_rate': LEARNING_RATE,
                'weight_decay': WEIGHT_DECAY,
                'epochs_max': EPOCHS_MAX,
                'patience': PATIENCE,
                'val_fraction': VAL_FRACTION,
                'n_inner_splits': N_INNER_SPLITS,
                'seed': SEED
            }
        )
        
        # 3. Calcola metriche aggregate
        print(f"\n{'='*70}")
        print("  RISULTATI AGGREGATI")
        print(f"{'='*70}")
        
        summary_df = compute_aggregate_metrics(results_df)
        print(f"\n{summary_df.to_string(index=False)}")
        
        # Salva metriche
        summary_df.to_csv(OUTPUT_DIR / 'model_comparison_summary.csv', index=False)
        print(f"\n  ✓ Metriche salvate in model_comparison_summary.csv")
        
        # 4. Salva risultati con architettura modello
        print(f"\n{'─'*70}")
        print("  FASE 3: Salvataggio risultati e configurazione modello")
        print(f"{'─'*70}")
        
        # Estrai metriche per lo storico (dal modello MLP)
        mlp_row = summary_df[summary_df['Model'] == 'MLP (multi-feature)']
        metrics = {
            'R2': float(mlp_row['R²'].values[0]),
            'MAE': float(mlp_row['MAE'].values[0]),
            'RMSE': float(mlp_row['RMSE'].values[0])
        }
        
        save_results(
            results_df=results_df,
            importances=all_importances,
            fold_curves=fold_curves,
            output_dir=OUTPUT_DIR,
            features=FEATURES,
            model_config={
                'hidden1': HIDDEN_1,
                'hidden2': HIDDEN_2,
                'dropout': DROPOUT,
                'learning_rate': LEARNING_RATE,
                'weight_decay': WEIGHT_DECAY,
                'epochs_max': EPOCHS_MAX,
                'patience': PATIENCE,
                'val_fraction': VAL_FRACTION,
                'n_inner_splits': N_INNER_SPLITS,
                'seed': SEED
            },
            metrics=metrics
        )
        
        # 5. Genera visualizzazioni
        print(f"\n{'─'*70}")
        print("  FASE 4: Generazione visualizzazioni")
        print(f"{'─'*70}")
        
        plot_all_results(
            results_df=results_df,
            importances=all_importances,
            fold_curves=fold_curves,
            output_dir=OUTPUT_DIR,
            features=FEATURES
        )
        
        # 6. Report finale
        print(f"\n{'='*70}")
        print("  ANALISI COMPLETA")
        print(f"{'='*70}")
        print(f"\n  Output salvati in: {OUTPUT_DIR}")
        print(f"\n  File generati:")
        print(f"    • loocv_predictions.csv              — predizioni per paziente")
        print(f"    • model_comparison_summary.csv       — R², MAE, RMSE aggregati")
        print(f"    • feature_importances.csv            — importance per feature")
        print(f"    • config_summary.json                — configurazione modello attuale")
        print(f"    • model_architectures_history.csv   — storico architetture (CSV)")
        print(f"    • plot_actual_vs_predicted.png       — scatter actual vs predicted")
        print(f"    • plot_bland_altman.png              — Bland-Altman")
        print(f"    • plot_per_patient_errors.png    — errori per paziente")
        print(f"    • plot_feature_importance.png    — permutation importance")
        print(f"    • training_curves/               — loss curves per fold")
        print(f"\n{'='*70}\n")
        
    except FileNotFoundError as e:
        print(f"\n✗ Errore: {e}")
        print("  → Assicurati che il file CSV esista e il percorso sia corretto")
        print("  → Esegui prima improved_fvc_prediction.py per generare il dataset")
    except Exception as e:
        print(f"\n✗ Errore durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()