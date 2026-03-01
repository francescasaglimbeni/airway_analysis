"""
Fine-tuning script per il best model di un dataset specifico.
Esegue grid search sui parametri più rilevanti del modello migliore.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import warnings
from itertools import product
from tqdm import tqdm
import json
warnings.filterwarnings('ignore')

from data.preprocessing import load_and_preprocess_data
from data.splits import run_loocv
from evaluation.metrics import compute_aggregate_metrics

# ============================================================================
# CONFIGURAZIONI PATHS
# ============================================================================

BASE_DIR = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\OSIC_metrics_validation")
UNIFIED_DIR = BASE_DIR / "unified_prediction"

# Dataset disponibili e relativi best models (da overall_summary.csv)
DATASET_CONFIGS = {
    'strict': {
        'path': UNIFIED_DIR / "dataset_strict.csv",
        'best_model': 'Lasso',
        'best_r2': 0.7769,
        'n_patients': 22
    },
    'balanced': {
        'path': UNIFIED_DIR / "dataset_balanced.csv",
        'best_model': 'Ensemble',
        'best_r2': 0.5354,
        'n_patients': 32
    },
    'traditional_only': {
        'path': UNIFIED_DIR / "dataset_traditional_only.csv",
        'best_model': 'Ensemble',
        'best_r2': 0.5354,
        'n_patients': 32
    },
    'both_targets': {
        'path': UNIFIED_DIR / "dataset_both_targets.csv",
        'best_model': 'Ensemble',
        'best_r2': 0.5354,
        'n_patients': 32
    },
    'all': {
        'path': UNIFIED_DIR / "dataset_all.csv",
        'best_model': 'XGBoost',
        'best_r2': 0.5024,
        'n_patients': 34
    }
}

# ============================================================================
# GRID SEARCH CONFIGURATIONS
# ============================================================================

# Configurazioni per Lasso (L1 regularization)
LASSO_GRID = {
    'lasso_alpha': [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
}

# Configurazioni per Ridge (L2 regularization)
RIDGE_GRID = {
    'ridge_alpha': [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
}

# Configurazioni per Random Forest
RF_GRID = {
    'rf_n_estimators': [50, 100, 200],
    'rf_max_depth': [2, 3, 4, 5],
    'rf_min_samples_split': [2, 5, 10],
    'rf_min_samples_leaf': [1, 2, 4]
}

# Configurazioni per XGBoost
XGB_GRID = {
    'xgb_n_estimators': [50, 100, 150, 200],
    'xgb_max_depth': [2, 3, 4, 5],
    'xgb_learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'xgb_reg_alpha': [0.0, 0.5, 1.0, 2.0, 3.0],  # L1 reg
    'xgb_reg_lambda': [0.0, 0.5, 1.0, 2.0, 3.0]   # L2 reg
}

# Configurazioni per Ensemble (Ridge + RF)
ENSEMBLE_GRID = {
    'ensemble_ridge_weight': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'ensemble_rf_weight': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],  # sarà calcolato come 1 - ridge_weight
    'ridge_alpha': [1.0, 2.0, 5.0, 7.0, 10.0],  # Ridge params
    'rf_n_estimators': [50, 100, 200],          # RF params
    'rf_max_depth': [2, 3, 4],
    'rf_min_samples_split': [2, 5, 10]
}

# ============================================================================
# FEATURES E TARGET
# ============================================================================

FEATURES = [
    'FVC_percent_week0',
    'mean_peripheral_branch_volume_mm3',
    'peripheral_branch_density',
    'mean_peripheral_diameter_mm',
    'central_to_peripheral_diameter_ratio',
    'mean_lung_density_HU',
    'histogram_entropy',
]

TARGET = 'FVC_percent_week52'

# Configurazioni base (da usare quando non specificate)
BASE_CONFIG = {
    'hidden1': 16,
    'hidden2': 8,
    'dropout': 0.2,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'epochs_max': 500,
    'patience': 100,
    'val_fraction': 0.20,
    'n_inner_splits': 10,
    'seed': 42,
    'ridge_alpha': 5.0,
    'lasso_alpha': 0.5,
    'rf_n_estimators': 100,
    'rf_max_depth': 2,
    'rf_min_samples_split': 5,
    'rf_min_samples_leaf': 2,
    'xgb_n_estimators': 100,
    'xgb_max_depth': 2,
    'xgb_learning_rate': 0.1,
    'xgb_reg_alpha': 1.0,
    'xgb_reg_lambda': 1.0,
    'ensemble_ridge_weight': 0.7,
    'ensemble_rf_weight': 0.3,
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# FUNZIONI PRINCIPALI
# ============================================================================

def generate_param_combinations(grid_dict, model_type):
    """
    Genera tutte le combinazioni di parametri per il grid search.
    
    Args:
        grid_dict: Dizionario con liste di valori per ogni parametro
        model_type: Tipo di modello ('Lasso', 'Ridge', 'RF', 'XGBoost', 'Ensemble')
    
    Returns:
        Lista di dizionari, ciascuno con una combinazione di parametri
    """
    if model_type == 'Ensemble':
        # Per Ensemble, garantisce che ridge_weight + rf_weight = 1
        combinations = []
        weights = grid_dict['ensemble_ridge_weight']
        ridge_alphas = grid_dict['ridge_alpha']
        rf_estimators = grid_dict['rf_n_estimators']
        rf_depths = grid_dict['rf_max_depth']
        rf_splits = grid_dict['rf_min_samples_split']
        
        for w_ridge in weights:
            w_rf = round(1.0 - w_ridge, 2)
            for ridge_alpha in ridge_alphas:
                for n_est in rf_estimators:
                    for depth in rf_depths:
                        for split in rf_splits:
                            combinations.append({
                                'ensemble_ridge_weight': w_ridge,
                                'ensemble_rf_weight': w_rf,
                                'ridge_alpha': ridge_alpha,
                                'rf_n_estimators': n_est,
                                'rf_max_depth': depth,
                                'rf_min_samples_split': split,
                                'rf_min_samples_leaf': 2  # fisso
                            })
        return combinations
    else:
        # Per altri modelli, genera prodotto cartesiano
        keys = list(grid_dict.keys())
        values = [grid_dict[k] for k in keys]
        combinations = []
        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))
        return combinations


def extract_model_metric(summary_df, model_keyword):
    """
    Estrae le metriche per un modello specifico dal summary_df.
    
    Args:
        summary_df: DataFrame con colonne ['Model', 'R²', 'MAE', 'RMSE']
        model_keyword: Parola chiave per identificare il modello
    
    Returns:
        dict con 'r2', 'mae', 'rmse' o None se non trovato
    """
    # Filtra per modello
    if model_keyword == 'Lasso':
        model_row = summary_df[summary_df['Model'].str.contains('Lasso', case=False, na=False)]
    elif model_keyword == 'Ridge':
        model_row = summary_df[summary_df['Model'].str.contains('Ridge', case=False, na=False)]
        # Escludi "Ensemble (Ridge+RF)"
        model_row = model_row[~model_row['Model'].str.contains('Ensemble', case=False, na=False)]
    elif model_keyword == 'RF' or model_keyword == 'RandomForest':
        model_row = summary_df[summary_df['Model'].str.contains('Random Forest', case=False, na=False)]
    elif model_keyword == 'XGBoost':
        model_row = summary_df[summary_df['Model'].str.contains('XGBoost', case=False, na=False)]
    elif model_keyword == 'Ensemble':
        model_row = summary_df[summary_df['Model'].str.contains('Ensemble', case=False, na=False)]
    else:
        model_row = pd.DataFrame()
    
    if model_row.empty:
        return None
    
    row = model_row.iloc[0]
    return {
        'r2': float(row['R²']),
        'mae': float(row['MAE']),
        'rmse': float(row['RMSE'])
    }


def run_single_configuration(dataset_path, config_params, base_config):
    """
    Esegue LOOCV con una singola configurazione di parametri.
    
    Args:
        dataset_path: Path al dataset CSV
        config_params: Dizionario con parametri specifici del modello
        base_config: Configurazione base da estendere
    
    Returns:
        dict con metriche {'r2', 'mae', 'rmse'} o None in caso di errore
    """
    try:
        # Carica e preprocessa
        df_clean = load_and_preprocess_data(
            input_path=dataset_path,
            features=FEATURES,
            target=TARGET
        )
        
        # Merge config
        full_config = {**base_config, **config_params}
        
        # LOOCV
        results_df, _, _ = run_loocv(
            df=df_clean,
            features=FEATURES,
            target=TARGET,
            device=DEVICE,
            config=full_config
        )
        
        # Calcola metriche
        summary_df = compute_aggregate_metrics(results_df, config=full_config)
        
        return summary_df
        
    except Exception as e:
        print(f"\n  ✗ Errore: {e}")
        return None


def fine_tune_model(dataset_name, model_type, output_dir, max_configs=None):
    """
    Esegue fine-tuning del modello specificato sul dataset scelto.
    
    Args:
        dataset_name: Nome del dataset ('strict', 'balanced', ecc.)
        model_type: Tipo di modello ('Lasso', 'Ridge', 'RF', 'XGBoost', 'Ensemble')
        output_dir: Directory dove salvare i risultati
        max_configs: Numero massimo di configurazioni da testare (None = tutte)
    
    Returns:
        DataFrame con tutti i risultati
    """
    print(f"\n{'='*80}")
    print(f"  FINE-TUNING: {model_type} su {dataset_name}")
    print(f"{'='*80}")
    
    # Verifica dataset
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Dataset '{dataset_name}' non riconosciuto. Disponibili: {list(DATASET_CONFIGS.keys())}")
    
    dataset_info = DATASET_CONFIGS[dataset_name]
    dataset_path = dataset_info['path']
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset non trovato: {dataset_path}")
    
    print(f"  Dataset: {dataset_path}")
    print(f"  N pazienti: {dataset_info['n_patients']}")
    print(f"  Best R² precedente: {dataset_info['best_r2']:.4f}")
    
    # Seleziona grid appropriato
    if model_type == 'Lasso':
        grid = LASSO_GRID
    elif model_type == 'Ridge':
        grid = RIDGE_GRID
    elif model_type in ['RF', 'RandomForest']:
        grid = RF_GRID
        model_type = 'RF'
    elif model_type == 'XGBoost':
        grid = XGB_GRID
    elif model_type == 'Ensemble':
        grid = ENSEMBLE_GRID
    else:
        raise ValueError(f"Tipo di modello '{model_type}' non supportato")
    
    # Genera combinazioni
    param_combinations = generate_param_combinations(grid, model_type)
    
    # Limita numero di configurazioni se richiesto
    if max_configs and len(param_combinations) > max_configs:
        print(f"  ⚠ Troppe configurazioni ({len(param_combinations)}), limitando a {max_configs}")
        # Campionamento casuale
        np.random.seed(42)
        indices = np.random.choice(len(param_combinations), max_configs, replace=False)
        param_combinations = [param_combinations[i] for i in indices]
    
    print(f"  Configurazioni da testare: {len(param_combinations)}")
    print()
    
    # Esegui grid search
    all_results = []
    
    for idx, params in enumerate(tqdm(param_combinations, desc="  Grid Search")):
        # Esegui configurazione
        summary_df = run_single_configuration(dataset_path, params, BASE_CONFIG)
        
        if summary_df is None:
            continue
        
        # Estrai metriche per il modello target
        metrics = extract_model_metric(summary_df, model_type)
        
        if metrics is None:
            continue
        
        # Salva risultato
        result = {
            'config_id': idx + 1,
            **params,
            'r2': metrics['r2'],
            'mae': metrics['mae'],
            'rmse': metrics['rmse']
        }
        all_results.append(result)
    
    # Converti a DataFrame
    results_df = pd.DataFrame(all_results)
    
    if results_df.empty:
        print("\n  ✗ Nessun risultato valido ottenuto")
        return None
    
    # Ordina per R² decrescente
    results_df = results_df.sort_values('r2', ascending=False).reset_index(drop=True)
    
    # Salva risultati
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"fine_tuning_{dataset_name}_{model_type}.csv"
    results_df.to_csv(results_path, index=False)
    
    # Identifica best config
    best_config = results_df.iloc[0]
    
    print(f"\n{'─'*80}")
    print(f"  RISULTATI FINE-TUNING")
    print(f"{'─'*80}")
    print(f"  Configurazioni testate: {len(results_df)}")
    print(f"  Best R²: {best_config['r2']:.4f} (MAE={best_config['mae']:.2f}, RMSE={best_config['rmse']:.2f})")
    print(f"  Miglioramento rispetto a baseline: {(best_config['r2'] - dataset_info['best_r2']):.4f}")
    
    # Stampa best config
    param_cols = [col for col in results_df.columns if col not in ['config_id', 'r2', 'mae', 'rmse']]
    print(f"\n  Best Configuration:")
    for param in param_cols:
        print(f"    {param}: {best_config[param]}")
    
    # Salva best config come JSON
    best_config_dict = best_config[param_cols].to_dict()
    best_config_path = output_dir / f"best_config_{dataset_name}_{model_type}.json"
    with open(best_config_path, 'w') as f:
        json.dump(best_config_dict, f, indent=2)
    
    print(f"\n  ✓ Risultati salvati in: {output_dir}/")
    print(f"    - {results_path.name}")
    print(f"    - {best_config_path.name}")
    
    # Crea plot di visualizzazione
    create_tuning_plots(results_df, model_type, dataset_name, output_dir)
    
    return results_df


def create_tuning_plots(results_df, model_type, dataset_name, output_dir):
    """
    Crea plot di visualizzazione dei risultati del fine-tuning.
    """
    print(f"\n  Creazione plot di visualizzazione...")
    
    # Top 10 configurazioni
    top_k = min(10, len(results_df))
    top_results = results_df.head(top_k)
    
    # Plot 1: Top K configurazioni
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Top {top_k} Configurations - {model_type} on {dataset_name}', 
                 fontsize=14, fontweight='bold')
    
    x_labels = [f"Config {i+1}" for i in range(top_k)]
    
    # R²
    axes[0].bar(range(top_k), top_results['r2'], color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Configuration')
    axes[0].set_ylabel('R²')
    axes[0].set_title(f'R² Score')
    axes[0].set_xticks(range(top_k))
    axes[0].set_xticklabels(x_labels, rotation=45, ha='right')
    axes[0].grid(axis='y', alpha=0.3)
    
    # MAE
    axes[1].bar(range(top_k), top_results['mae'], color='coral', alpha=0.7)
    axes[1].set_xlabel('Configuration')
    axes[1].set_ylabel('MAE')
    axes[1].set_title(f'Mean Absolute Error')
    axes[1].set_xticks(range(top_k))
    axes[1].set_xticklabels(x_labels, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3)
    
    # RMSE
    axes[2].bar(range(top_k), top_results['rmse'], color='lightgreen', alpha=0.7)
    axes[2].set_xlabel('Configuration')
    axes[2].set_ylabel('RMSE')
    axes[2].set_title(f'Root Mean Squared Error')
    axes[2].set_xticks(range(top_k))
    axes[2].set_xticklabels(x_labels, rotation=45, ha='right')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / f"top_configs_{dataset_name}_{model_type}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Distribuzione parametri per top configs (solo se applicabile)
    param_cols = [col for col in results_df.columns 
                  if col not in ['config_id', 'r2', 'mae', 'rmse']]
    
    if param_cols:
        n_params = len(param_cols)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        fig.suptitle(f'Parameter Distributions (Top {top_k}) - {model_type} on {dataset_name}', 
                     fontsize=14, fontweight='bold')
        
        if n_params == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, param in enumerate(param_cols):
            ax = axes[idx]
            values = top_results[param]
            
            if values.dtype in [np.float64, np.float32, np.int64, np.int32]:
                # Numerico: istogramma o boxplot
                if len(values.unique()) > 5:
                    ax.hist(values, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
                    ax.set_xlabel(param)
                    ax.set_ylabel('Frequency')
                else:
                    value_counts = values.value_counts().sort_index()
                    ax.bar(range(len(value_counts)), value_counts.values, 
                           tick_label=value_counts.index, color='skyblue', alpha=0.7)
                    ax.set_xlabel(param)
                    ax.set_ylabel('Count')
            else:
                # Categorico
                value_counts = values.value_counts()
                ax.bar(range(len(value_counts)), value_counts.values, 
                       tick_label=value_counts.index, color='skyblue', alpha=0.7)
                ax.set_xlabel(param)
                ax.set_ylabel('Count')
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            ax.set_title(f'{param}')
            ax.grid(axis='y', alpha=0.3)
        
        # Nascondi assi non usati
        for idx in range(n_params, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plot_path = output_dir / f"param_distributions_{dataset_name}_{model_type}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"    ✓ Plot salvati")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Main entry point per lo script di fine-tuning.
    """
    print("\n" + "="*80)
    print("  FINE-TUNING BEST MODEL")
    print("="*80)
    
    # Mostra dataset disponibili e relativi best models
    print("\n  Dataset disponibili:")
    for name, info in DATASET_CONFIGS.items():
        print(f"    [{name}]")
        print(f"      Best model: {info['best_model']}")
        print(f"      R²: {info['best_r2']:.4f}")
        print(f"      N pazienti: {info['n_patients']}")
    
    # Richiedi input utente
    print("\n" + "─"*80)
    dataset_name = input("  Scegli dataset (strict/balanced/traditional_only/both_targets/all): ").strip()
    
    if dataset_name not in DATASET_CONFIGS:
        print(f"\n  ✗ Dataset '{dataset_name}' non valido")
        return
    
    # Ottieni info dataset
    dataset_info = DATASET_CONFIGS[dataset_name]
    suggested_model = dataset_info['best_model']
    
    print(f"\n  Best model per '{dataset_name}': {suggested_model}")
    model_type = input(f"  Modello da fine-tunare [{suggested_model}/Lasso/Ridge/RF/XGBoost/Ensemble]: ").strip()
    
    if not model_type:
        model_type = suggested_model
    
    # Richiedi limite configurazioni
    max_configs_input = input("\n  Numero massimo di configurazioni da testare [vuoto=tutte]: ").strip()
    max_configs = int(max_configs_input) if max_configs_input else None
    
    # Output directory
    output_dir = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\validation_test_models\fine_tuning_results")
    
    # Esegui fine-tuning
    results_df = fine_tune_model(
        dataset_name=dataset_name,
        model_type=model_type,
        output_dir=output_dir,
        max_configs=max_configs
    )
    
    if results_df is not None:
        print(f"\n{'='*80}")
        print("  COMPLETATO")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
