"""
Grid search COMPATTA per dataset piccolo (31 pazienti).
Solo le configurazioni più promettenti basate su best practices per small datasets.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import warnings
from itertools import product
from datetime import datetime
import json
warnings.filterwarnings('ignore')

from data.preprocessing import load_and_preprocess_data
from data.splits import run_loocv
from evaluation.metrics import compute_aggregate_metrics

# Paths
INPUT_CSV = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\OSIC_metrics_validation\improved_prediction\all_data_dataset.csv")
OUTPUT_DIR = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\validation_test_models\quick_search_results")

FEATURES = [
    'mean_peripheral_branch_volume_mm3',
    'peripheral_branch_density',
    'mean_peripheral_diameter_mm',
    'central_to_peripheral_diameter_ratio',
    'mean_lung_density_HU',
    'histogram_entropy',
]
TARGET = 'FVC_percent_week52'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parametri fissi
SEED = 42
EPOCHS_MAX = 500
PATIENCE = 100
VAL_FRACTION = 0.20
N_INNER_SPLITS = 10

# ============================================================================
# CONFIGURAZIONI COMPATTE - SOLO LE PIÙ PROMETTENTI
# ============================================================================

# --- MLP: 3 configurazioni essenziali ---
MLP_CONFIGS = [
    # (h1, h2, dropout, lr, wd)
    (16, 8, 0.2, 1e-3, 1e-4),   # baseline standard
    (16, 8, 0.3, 1e-3, 1e-3),   # baseline + forte regolarizzazione
    (20, 10, 0.3, 5e-4, 1e-3),  # media + lenta + reg forte
]

# --- RIDGE: 3 valori α chiave ---
RIDGE_CONFIGS = [1.0, 2.0, 5.0]

# --- LASSO: 2 valori α chiave ---
LASSO_CONFIGS = [0.2, 0.5]

# --- RANDOM FOREST: 3 configurazioni conservative ---
RF_CONFIGS = [
    # (n_estimators, max_depth, min_samples_split, min_samples_leaf)
    (100, 2, 5, 2),   # molto shallow
    (100, 3, 5, 2),   # baseline conservativo
    (100, 4, 5, 2),   # leggermente più profondo
]

# --- ENSEMBLE: 2 configurazioni pesi ---
ENSEMBLE_CONFIGS = [
    (0.6, 0.4),  # Ridge leggermente dominante (baseline)
    (0.7, 0.3),  # Ridge dominante
]


def save_checkpoint(all_results, tested_configs, checkpoint_file, state_file):
    """Salva checkpoint con risultati e configurazioni già testate"""
    # Salva risultati
    if all_results:
        pd.DataFrame(all_results).to_csv(checkpoint_file, index=False)
    
    # Salva stato (configurazioni già testate)
    with open(state_file, 'w') as f:
        json.dump({'tested_configs': list(tested_configs)}, f)


def load_checkpoint(checkpoint_file, state_file):
    """Carica checkpoint se esiste"""
    results = []
    tested_configs = set()
    
    if checkpoint_file.exists():
        try:
            df = pd.read_csv(checkpoint_file)
            results = df.to_dict('records')
            print(f"  ✓ Caricati {len(results)} risultati dal checkpoint")
        except Exception as e:
            print(f"  ⚠ Errore caricamento risultati: {e}")
    
    if state_file.exists():
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
                tested_configs = set(state.get('tested_configs', []))
            print(f"  ✓ Trovate {len(tested_configs)} configurazioni già testate")
        except Exception as e:
            print(f"  ⚠ Errore caricamento stato: {e}")
    
    return results, tested_configs


def get_config_signature(mlp_cfg, ridge_cfg, lasso_cfg, rf_cfg, ens_cfg):
    """Genera un identificatore unico per una configurazione"""
    return f"mlp_{mlp_cfg}_ridge_{ridge_cfg}_lasso_{lasso_cfg}_rf_{rf_cfg}_ens_{ens_cfg}"


def test_configuration(df_clean, config_id, mlp_cfg, ridge_cfg, lasso_cfg, rf_cfg, ens_cfg):
    """Testa una configurazione completa"""
    
    mlp_h1, mlp_h2, mlp_drop, mlp_lr, mlp_wd = mlp_cfg
    ridge_alpha = ridge_cfg
    lasso_alpha = lasso_cfg
    rf_n_est, rf_depth, rf_split, rf_leaf = rf_cfg
    ens_ridge_w, ens_rf_w = ens_cfg
    
    print(f"\n{'─'*70}")
    print(f"  CONFIG {config_id}")
    print(f"  MLP: {mlp_h1}-{mlp_h2}, drop={mlp_drop}, lr={mlp_lr:.0e}, wd={mlp_wd:.0e}")
    print(f"  Ridge: α={ridge_alpha} | Lasso: α={lasso_alpha}")
    print(f"  RF: n={rf_n_est}, d={rf_depth} | Ens: R={ens_ridge_w:.1f}, RF={ens_rf_w:.1f}")
    
    config = {
        'hidden1': mlp_h1, 'hidden2': mlp_h2, 'dropout': mlp_drop,
        'learning_rate': mlp_lr, 'weight_decay': mlp_wd,
        'epochs_max': EPOCHS_MAX, 'patience': PATIENCE,
        'val_fraction': VAL_FRACTION, 'n_inner_splits': N_INNER_SPLITS,
        'seed': SEED,
        'ridge_alpha': ridge_alpha, 'lasso_alpha': lasso_alpha,
        'rf_n_estimators': rf_n_est, 'rf_max_depth': rf_depth,
        'rf_min_samples_split': rf_split, 'rf_min_samples_leaf': rf_leaf,
        'ensemble_ridge_weight': ens_ridge_w, 'ensemble_rf_weight': ens_rf_w,
    }
    
    try:
        results_df, _, _ = run_loocv(df_clean, FEATURES, TARGET, DEVICE, config)
        summary_df = compute_aggregate_metrics(results_df)
        
        result = {
            'config_id': config_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'mlp_h1': mlp_h1, 'mlp_h2': mlp_h2, 'mlp_dropout': mlp_drop,
            'mlp_lr': mlp_lr, 'mlp_wd': mlp_wd,
            'ridge_alpha': ridge_alpha, 'lasso_alpha': lasso_alpha,
            'rf_n_estimators': rf_n_est, 'rf_max_depth': rf_depth,
            'rf_min_samples_split': rf_split, 'rf_min_samples_leaf': rf_leaf,
            'ensemble_ridge_weight': ens_ridge_w, 'ensemble_rf_weight': ens_rf_w,
        }
        
        # Estrai metriche per tutti i modelli
        for _, row in summary_df.iterrows():
            model_name = row['Model']
            prefix = model_name.split('(')[0].strip().lower().replace(' ', '_').replace('-', '_')
            result[f'{prefix}_r2'] = float(row['R²'])
            result[f'{prefix}_mae'] = float(row['MAE'])
            result[f'{prefix}_rmse'] = float(row['RMSE'])
        
        # Mostra metriche principali
        print(f"  ✓ MLP: R²={result.get('mlp_r2', 0):.4f}, MAE={result.get('mlp_mae', 0):.2f}")
        print(f"  ✓ Ens: R²={result.get('ensemble_r2', 0):.4f}, MAE={result.get('ensemble_mae', 0):.2f}")
        print(f"  ✓ Ridge: R²={result.get('ridge_r2', 0):.4f}, MAE={result.get('ridge_mae', 0):.2f}")
        print(f"  ✓ RF: R²={result.get('random_forest_r2', 0):.4f}, MAE={result.get('random_forest_mae', 0):.2f}")
        
        return result
    except Exception as e:
        print(f"  ✗ ERRORE: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("\n" + "="*70)
    print("  GRID SEARCH COMPATTA - CONFIGURAZIONI CHIAVE")
    print("  Dataset: 31 pazienti | 6 features | LOOCV")
    print(f"  Device: {DEVICE}")
    print("="*70)
    
    if torch.cuda.is_available():
        print(f"\n  🚀 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"\n  💻 CPU mode")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # File checkpoint
    checkpoint_file = OUTPUT_DIR / 'checkpoint_results.csv'
    state_file = OUTPUT_DIR / 'checkpoint_state.json'
    
    # Carica checkpoint se esiste
    print(f"\n{'─'*70}")
    print("  Controllo checkpoint...")
    print(f"{'─'*70}")
    all_results, tested_configs = load_checkpoint(checkpoint_file, state_file)
    
    if tested_configs:
        print(f"\n  🔄 RIPRESA DA CHECKPOINT")
        print(f"  ✓ {len(tested_configs)} configurazioni già completate")
        response = input(f"\n  ▶ Continuare da dove interrotto? (y per continuare, n per ricominciare): ")
        if response.lower() == 'n':
            all_results = []
            tested_configs = set()
            print("  ⚠ Checkpoint ignorato, ripartenza da zero")
    
    # Carica dati
    print(f"\n{'─'*70}")
    print("  Caricamento dati...")
    print(f"{'─'*70}")
    df_clean = load_and_preprocess_data(INPUT_CSV, FEATURES, TARGET)
    print(f"  ✓ {len(df_clean)} pazienti caricati")
    
    # Calcola totale configurazioni
    total = (len(MLP_CONFIGS) * len(RIDGE_CONFIGS) * len(LASSO_CONFIGS) * 
             len(RF_CONFIGS) * len(ENSEMBLE_CONFIGS))
    remaining = total - len(tested_configs)
    
    print(f"\n{'='*70}")
    print(f"  CONFIGURAZIONI TOTALI: {total}")
    if tested_configs:
        print(f"  GIÀ COMPLETATE: {len(tested_configs)}")
        print(f"  RIMANENTI: {remaining}")
    print(f"{'='*70}")
    print(f"  • MLP: {len(MLP_CONFIGS)} configurazioni")
    print(f"  • Ridge: {len(RIDGE_CONFIGS)} valori α")
    print(f"  • Lasso: {len(LASSO_CONFIGS)} valori α")
    print(f"  • Random Forest: {len(RF_CONFIGS)} configurazioni")
    print(f"  • Ensemble: {len(ENSEMBLE_CONFIGS)} combinazioni pesi")
    
    # Stima tempo (più conservativa)
    minutes_per_config = 3.5
    total_hours = (remaining * minutes_per_config) / 60
    
    print(f"\n  ⏱ TEMPO STIMATO (per configurazioni rimanenti):")
    print(f"     CPU: ~{total_hours:.1f} ore ({total_hours/24:.1f} giorni)")
    if torch.cuda.is_available():
        gpu_hours = total_hours / 2.5
        print(f"     GPU: ~{gpu_hours:.1f} ore ({gpu_hours/24:.1f} giorni)")
    
    print(f"\n  💾 Checkpoint automatico dopo ogni configurazione")
    print(f"  📊 Output directory: {OUTPUT_DIR.name}/")
    
    response = input(f"\n  ▶ Avviare ricerca? (y/n): ")
    if response.lower() != 'y':
        print("\n  Annullato.")
        return
    
    # Esegui ricerca
    config_id = 0
    configs_tested_this_session = 0
    start_time = datetime.now()
    
    print(f"\n{'#'*70}")
    if tested_configs:
        print("  RIPRESA RICERCA")
    else:
        print("  INIZIO RICERCA")
    print(f"{'#'*70}")
    
    try:
        for mlp_cfg in MLP_CONFIGS:
            for ridge_cfg in RIDGE_CONFIGS:
                for lasso_cfg in LASSO_CONFIGS:
                    for rf_cfg in RF_CONFIGS:
                        for ens_cfg in ENSEMBLE_CONFIGS:
                            config_id += 1
                            
                            # Genera signature per questa configurazione
                            config_sig = get_config_signature(mlp_cfg, ridge_cfg, lasso_cfg, rf_cfg, ens_cfg)
                            
                            # Skip se già testata
                            if config_sig in tested_configs:
                                continue
                            
                            configs_tested_this_session += 1
                            
                            # Progress update
                            completed = len(tested_configs) + configs_tested_this_session
                            if configs_tested_this_session > 1:
                                elapsed_h = (datetime.now() - start_time).total_seconds() / 3600
                                avg_time = elapsed_h / configs_tested_this_session
                                remaining_configs = remaining - configs_tested_this_session
                                remaining_h = avg_time * remaining_configs
                                eta = datetime.now() + pd.Timedelta(hours=remaining_h)
                                progress_pct = (completed / total) * 100
                                
                                print(f"\n{'#'*70}")
                                print(f"  PROGRESSO: {completed}/{total} ({progress_pct:.1f}%)")
                                print(f"  Sessione corrente: {configs_tested_this_session} nuove configurazioni")
                                print(f"  Tempo: {elapsed_h:.1f}h trascorse | ~{remaining_h:.1f}h rimanenti")
                                print(f"  ETA: {eta.strftime('%d/%m/%Y %H:%M')}")
                                print(f"{'#'*70}")
                            
                            result = test_configuration(df_clean, config_id, mlp_cfg, 
                                                       ridge_cfg, lasso_cfg, rf_cfg, ens_cfg)
                            
                            if result:
                                all_results.append(result)
                                tested_configs.add(config_sig)
                                
                                # Salva checkpoint dopo ogni configurazione
                                save_checkpoint(all_results, tested_configs, checkpoint_file, state_file)
                                print(f"  💾 Checkpoint salvato ({len(all_results)} configurazioni totali)")
    
    except KeyboardInterrupt:
        print(f"\n\n{'='*70}")
        print("  ⚠ INTERRUZIONE MANUALE")
        print(f"{'='*70}")
        print(f"  Configurazioni completate: {len(all_results)}")
        print(f"  Checkpoint salvato: puoi riprendere in seguito")
        print(f"{'='*70}\n")
        return
    
    # ========================================================================
    # ANALISI FINALE
    # ========================================================================
    
    if not all_results:
        print("\n✗ Nessun risultato disponibile!")
        return
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / 'results.csv', index=False)
    
    # Rimuovi checkpoint se completato tutto
    if len(tested_configs) >= total:
        if checkpoint_file.exists():
            checkpoint_file.unlink()
        if state_file.exists():
            state_file.unlink()
        print(f"\n  ✓ Ricerca completata! Checkpoint rimossi.")
    
    print(f"\n\n{'='*70}")
    print("  📊 ANALISI RISULTATI FINALI")
    print(f"{'='*70}")
    print(f"\n  ✓ Completate: {len(results_df)}/{total} configurazioni")
    
    # Analizza ogni modello
    models = [
        ('MLP', 'mlp'),
        ('Ensemble', 'ensemble'),
        ('Ridge', 'ridge'),
        ('Random Forest', 'random_forest'),
        ('Lasso', 'lasso'),
        ('Linear Regression', 'lr'),
    ]
    
    best_configs = {}
    
    for model_name, key in models:
        r2_col = f'{key}_r2'
        mae_col = f'{key}_mae'
        rmse_col = f'{key}_rmse'
        
        if r2_col not in results_df.columns:
            continue
        
        print(f"\n{'─'*70}")
        print(f"  {model_name.upper()}")
        print(f"{'─'*70}")
        
        # Migliore per R²
        best_idx = results_df[r2_col].idxmax()
        best = results_df.loc[best_idx]
        
        print(f"\n  🏆 Migliore configurazione:")
        print(f"      R² = {best[r2_col]:.4f}")
        print(f"      MAE = {best[mae_col]:.2f}%")
        print(f"      RMSE = {best[rmse_col]:.2f}%")
        print(f"      Config ID: {best['config_id']}")
        
        # Mostra parametri
        if key == 'mlp':
            print(f"      Architettura: {int(best['mlp_h1'])}-{int(best['mlp_h2'])}")
            print(f"      Dropout: {best['mlp_dropout']:.2f}")
            print(f"      Learning rate: {best['mlp_lr']:.0e}")
            print(f"      Weight decay: {best['mlp_wd']:.0e}")
        elif key == 'ridge':
            print(f"      Alpha: {best['ridge_alpha']}")
        elif key == 'lasso':
            print(f"      Alpha: {best['lasso_alpha']}")
        elif key == 'random_forest':
            print(f"      n_estimators: {int(best['rf_n_estimators'])}")
            print(f"      max_depth: {int(best['rf_max_depth'])}")
        elif key == 'ensemble':
            print(f"      Ridge weight: {best['ensemble_ridge_weight']:.2f}")
            print(f"      RF weight: {best['ensemble_rf_weight']:.2f}")
        
        # Statistiche
        print(f"\n  📈 Statistiche globali:")
        print(f"      R² medio: {results_df[r2_col].mean():.4f} (std: {results_df[r2_col].std():.4f})")
        print(f"      MAE medio: {results_df[mae_col].mean():.2f}% (std: {results_df[mae_col].std():.2f})")
        print(f"      Range R²: [{results_df[r2_col].min():.4f}, {results_df[r2_col].max():.4f}]")
        
        # Salva best config
        best_configs[key] = {
            'model': model_name,
            'config_id': int(best['config_id']),
            'performance': {
                'R2': float(best[r2_col]),
                'MAE': float(best[mae_col]),
                'RMSE': float(best[rmse_col]),
            }
        }
        
        # Aggiungi parametri
        if key == 'mlp':
            best_configs[key]['parameters'] = {
                'h1': int(best['mlp_h1']),
                'h2': int(best['mlp_h2']),
                'dropout': float(best['mlp_dropout']),
                'lr': float(best['mlp_lr']),
                'wd': float(best['mlp_wd']),
            }
        elif key in ['ridge', 'lasso']:
            best_configs[key]['parameters'] = {
                'alpha': float(best[f'{key}_alpha'])
            }
        elif key == 'random_forest':
            best_configs[key]['parameters'] = {
                'n_estimators': int(best['rf_n_estimators']),
                'max_depth': int(best['rf_max_depth']),
                'min_samples_split': int(best['rf_min_samples_split']),
                'min_samples_leaf': int(best['rf_min_samples_leaf']),
            }
        elif key == 'ensemble':
            best_configs[key]['parameters'] = {
                'ridge_weight': float(best['ensemble_ridge_weight']),
                'rf_weight': float(best['ensemble_rf_weight']),
            }
    
    # Salva best configs
    with open(OUTPUT_DIR / 'best_configs.json', 'w') as f:
        json.dump(best_configs, f, indent=2)
    
    # Confronto finale
    print(f"\n{'='*70}")
    print("  🏆 CONFRONTO FINALE - MIGLIORI MODELLI")
    print(f"{'='*70}\n")
    
    comparison = []
    for key, cfg in best_configs.items():
        comparison.append({
            'Model': cfg['model'],
            'R²': cfg['performance']['R2'],
            'MAE': cfg['performance']['MAE'],
            'RMSE': cfg['performance']['RMSE'],
        })
    
    comp_df = pd.DataFrame(comparison).sort_values('R²', ascending=False)
    print(comp_df.to_string(index=False))
    comp_df.to_csv(OUTPUT_DIR / 'model_comparison.csv', index=False)
    
    # Vincitore
    winner = comp_df.iloc[0]
    print(f"\n{'='*70}")
    print(f"  🥇 VINCITORE ASSOLUTO: {winner['Model']}")
    print(f"{'='*70}")
    print(f"      R² = {winner['R²']:.4f}")
    print(f"      MAE = {winner['MAE']:.2f}%")
    print(f"      RMSE = {winner['RMSE']:.2f}%")
    print(f"{'='*70}")
    
    # Tempo finale
    total_time = (datetime.now() - start_time).total_seconds() / 3600
    print(f"\n  ⏱ Tempo totale esecuzione: {total_time:.2f} ore")
    print(f"\n  📁 Risultati salvati in: {OUTPUT_DIR}/")
    print(f"\n  File generati:")
    print(f"    • results.csv           — {len(results_df)} configurazioni testate")
    print(f"    • best_configs.json     — migliori config per ogni modello")
    print(f"    • model_comparison.csv  — confronto finale modelli")
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
