"""
TEST MODELLI SU TUTTI I DATASET DISPONIBILI - WEEK52 PREDICTION
Testa la predizione di FVC_percent_week52 su tutti i dataset provenienti da:
- improved_fvc_prediction.py
- improved_prediction_decline.py

Confronta le performance per identificare quale dataset è più predittivo.
"""

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

# ============================================================================
# CONFIGURAZIONE PATHS
# ============================================================================

BASE_DIR = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\OSIC_metrics_validation")

# Dataset da improved_prediction/
IMPROVED_PRED_DIR = BASE_DIR / "improved_prediction"
DATASETS_IMPROVED_PRED = {
    'quality_strict': IMPROVED_PRED_DIR / "quality_strict_dataset.csv",
    'quality_balanced': IMPROVED_PRED_DIR / "quality_balanced_dataset.csv",
    'all_data': IMPROVED_PRED_DIR / "all_data_dataset.csv"
}

# Dataset da improved_prediction_decline/
IMPROVED_DECLINE_DIR = BASE_DIR / "improved_prediction_decline"
DATASETS_IMPROVED_DECLINE = {
    'traditional_decline': IMPROVED_DECLINE_DIR / "02_dataset_traditional.csv",
    # NOTA: 04_dataset_direct_decline.csv NON ha FVC_percent_week52, 
    # ha solo FVC_annual_decline_direct quindi non lo includiamo
}

# Output
OUTPUT_DIR = Path(
    r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline"
    r"\validation_test_models\week52_all_datasets_comparison"
)

# ============================================================================
# FEATURES E TARGET
# ============================================================================

FEATURES = [
    'mean_peripheral_branch_volume_mm3',
    'peripheral_branch_density',
    'mean_peripheral_diameter_mm',
    'central_to_peripheral_diameter_ratio',
    'mean_lung_density_HU',
    'histogram_entropy',
]

TARGET = 'FVC_percent_week52'

# ============================================================================
# HYPERPARAMETERS MLP
# ============================================================================

HIDDEN_1 = 16
HIDDEN_2 = 8
DROPOUT = 0.2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS_MAX = 500
PATIENCE = 100
VAL_FRACTION = 0.20
N_INNER_SPLITS = 10
SEED = 42

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Stili grafici
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

# ============================================================================
# FUNZIONI PRINCIPALI
# ============================================================================

def test_single_dataset(dataset_name, dataset_path, output_subdir):
    """
    Testa un singolo dataset e salva i risultati in una sottodirectory
    """
    print(f"\n{'='*80}")
    print(f"  TESTING DATASET: {dataset_name}")
    print(f"{'='*80}")
    print(f"  Path: {dataset_path}")
    
    # Verifica esistenza
    if not dataset_path.exists():
        print(f"  ✗ File non trovato, SKIP")
        return None
    
    try:
        # 1. Carica e preprocessa
        print(f"\n  [1/4] Caricamento e preprocessing...")
        df_clean = load_and_preprocess_data(
            input_path=dataset_path,
            features=FEATURES,
            target=TARGET
        )
        
        n_patients = len(df_clean)
        print(f"        ✓ Dataset pulito: {n_patients} pazienti")
        
        if n_patients < 5:
            print(f"        ✗ Troppo pochi pazienti (< 5), SKIP")
            return None
        
        # 2. Esegui LOOCV
        print(f"\n  [2/4] Esecuzione LOOCV ({n_patients} fold)...")
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
        
        # 3. Calcola metriche
        print(f"\n  [3/4] Calcolo metriche aggregate...")
        summary_df = compute_aggregate_metrics(results_df)
        
        # Estrai metriche MLP
        mlp_row = summary_df[summary_df['Model'] == 'MLP (multi-feature)']
        r2 = float(mlp_row['R²'].values[0])
        mae = float(mlp_row['MAE'].values[0])
        rmse = float(mlp_row['RMSE'].values[0])
        
        print(f"        ✓ R² = {r2:.3f}, MAE = {mae:.2f}, RMSE = {rmse:.2f}")
        
        # 4. Salva risultati
        print(f"\n  [4/4] Salvataggio risultati...")
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        results_df.to_csv(output_subdir / 'loocv_predictions.csv', index=False)
        summary_df.to_csv(output_subdir / 'model_summary.csv', index=False)
        
        # Converti importances da lista a DataFrame
        if all_importances:
            importance_df = pd.DataFrame(all_importances)
            importance_df.to_csv(output_subdir / 'feature_importances.csv', index=False)
        else:
            importance_df = pd.DataFrame()  # DataFrame vuoto se non ci sono importances
        
        # 5. Genera plot
        plot_all_results(
            results_df=results_df,
            importances=all_importances,
            fold_curves=fold_curves,
            output_dir=output_subdir,
            features=FEATURES
        )
        
        print(f"        ✓ Risultati salvati in: {output_subdir.name}/")
        
        return {
            'dataset_name': dataset_name,
            'n_patients': n_patients,
            'R2': r2,
            'MAE': mae,
            'RMSE': rmse,
            'predictions_df': results_df,
            'summary_df': summary_df,
            'importances_df': importance_df
        }
        
    except Exception as e:
        print(f"  ✗ ERRORE durante test: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_comparison_plots(all_results, output_dir):
    """
    Crea plot comparativi tra tutti i dataset testati
    """
    print(f"\n{'='*80}")
    print("  CREAZIONE PLOT COMPARATIVI")
    print(f"{'='*80}")
    
    if len(all_results) < 2:
        print("  ⚠ Troppo pochi dataset per confronto")
        return
    
    # Prepara dati
    dataset_names = [r['dataset_name'] for r in all_results]
    r2_scores = [r['R2'] for r in all_results]
    mae_scores = [r['MAE'] for r in all_results]
    rmse_scores = [r['RMSE'] for r in all_results]
    n_patients = [r['n_patients'] for r in all_results]
    
    # -------------------------------------------------------------------------
    # 1. Barplot comparativo delle metriche
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Confronto Performance su Tutti i Dataset - Week52 Prediction', 
                 fontsize=14, fontweight='bold')
    
    # R²
    ax = axes[0]
    bars = ax.bar(range(len(dataset_names)), r2_scores, 
                  color=sns.color_palette("viridis", len(dataset_names)))
    ax.set_xticks(range(len(dataset_names)))
    ax.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax.set_ylabel('R² Score')
    ax.set_title('R² Score per Dataset')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    # Aggiungi valori sopra le barre
    for i, (bar, val, n) in enumerate(zip(bars, r2_scores, n_patients)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}\n(n={n})', ha='center', va='bottom', fontsize=9)
    
    # MAE
    ax = axes[1]
    bars = ax.bar(range(len(dataset_names)), mae_scores,
                  color=sns.color_palette("viridis", len(dataset_names)))
    ax.set_xticks(range(len(dataset_names)))
    ax.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax.set_ylabel('MAE (%)')
    ax.set_title('MAE per Dataset')
    ax.grid(axis='y', alpha=0.3)
    
    for i, (bar, val, n) in enumerate(zip(bars, mae_scores, n_patients)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.2f}\n(n={n})', ha='center', va='bottom', fontsize=9)
    
    # RMSE
    ax = axes[2]
    bars = ax.bar(range(len(dataset_names)), rmse_scores,
                  color=sns.color_palette("viridis", len(dataset_names)))
    ax.set_xticks(range(len(dataset_names)))
    ax.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax.set_ylabel('RMSE (%)')
    ax.set_title('RMSE per Dataset')
    ax.grid(axis='y', alpha=0.3)
    
    for i, (bar, val, n) in enumerate(zip(bars, rmse_scores, n_patients)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.2f}\n(n={n})', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_metrics_barplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Salvato: comparison_metrics_barplot.png")
    
    # -------------------------------------------------------------------------
    # 2. Scatter plot: R² vs n_patients
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(n_patients, r2_scores, s=200, alpha=0.7,
                        c=range(len(dataset_names)), cmap='viridis')
    
    # Aggiungi labels
    for i, name in enumerate(dataset_names):
        ax.annotate(name, (n_patients[i], r2_scores[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Numero Pazienti', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('R² vs Dimensione Dataset', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='R²=0')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_r2_vs_size.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Salvato: comparison_r2_vs_size.png")
    
    # -------------------------------------------------------------------------
    # 3. Feature importance comparison (heatmap)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Crea matrice di importance
    importance_matrix = []
    for result in all_results:
        imp_df = result['importances_df']
        # Media delle importance per feature
        avg_importance = imp_df.groupby('Feature')['Importance_Std'].mean()
        # Assicurati che abbia tutte le feature nell'ordine corretto
        avg_importance = avg_importance.reindex(FEATURES, fill_value=0)
        importance_matrix.append(avg_importance.values)
    
    importance_matrix = np.array(importance_matrix)
    
    # Crea heatmap
    sns.heatmap(importance_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=[f.replace('_', ' ') for f in FEATURES],
                yticklabels=dataset_names,
                cbar_kws={'label': 'Feature Importance (std)'}, ax=ax)
    
    ax.set_title('Confronto Feature Importance tra Dataset', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_feature_importance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Salvato: comparison_feature_importance_heatmap.png")
    
    # -------------------------------------------------------------------------
    # 4. Box-plot degli errori per dataset
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    
    all_errors = []
    labels = []
    
    for result in all_results:
        df = result['predictions_df']
        mlp_data = df[df['Model'] == 'MLP (multi-feature)']
        errors = mlp_data['Error'].values
        all_errors.append(errors)
        labels.append(f"{result['dataset_name']}\n(n={result['n_patients']})")
    
    bp = ax.boxplot(all_errors, labels=labels, patch_artist=True)
    
    # Colora boxes
    colors = sns.color_palette("viridis", len(all_errors))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Errore di Predizione (%)', fontsize=12)
    ax.set_title('Distribuzione Errori per Dataset', fontsize=14, fontweight='bold')
    ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_error_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Salvato: comparison_error_distributions.png")


def create_summary_table(all_results, output_dir):
    """
    Crea tabella riassuntiva con tutte le metriche
    """
    print(f"\n{'='*80}")
    print("  CREAZIONE TABELLA RIASSUNTIVA")
    print(f"{'='*80}")
    
    summary_data = []
    
    for result in all_results:
        summary_data.append({
            'Dataset': result['dataset_name'],
            'N_Patients': result['n_patients'],
            'R2': result['R2'],
            'MAE': result['MAE'],
            'RMSE': result['RMSE'],
            'MAE/RMSE_ratio': result['MAE'] / result['RMSE'] if result['RMSE'] > 0 else np.nan
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Ordina per R² decrescente
    df_summary = df_summary.sort_values('R2', ascending=False)
    
    # Aggiungi ranking
    df_summary.insert(0, 'Rank', range(1, len(df_summary) + 1))
    
    # Salva
    df_summary.to_csv(output_dir / 'overall_summary.csv', index=False)
    
    print(f"\n  RANKING DATASET PER R²:")
    print(f"  {'─'*80}")
    print(df_summary.to_string(index=False))
    print(f"  {'─'*80}")
    print(f"  ✓ Tabella salvata: overall_summary.csv")
    
    return df_summary


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("  TEST PREDIZIONE WEEK52 SU TUTTI I DATASET DISPONIBILI")
    print("="*80)
    print(f"  Target: {TARGET}")
    print(f"  Features: {len(FEATURES)}")
    print(f"  Device: {DEVICE}")
    print(f"  Output: {OUTPUT_DIR}")
    print("="*80)
    
    # Crea directory output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Raccogli tutti i dataset
    all_datasets = {}
    all_datasets.update(DATASETS_IMPROVED_PRED)
    all_datasets.update(DATASETS_IMPROVED_DECLINE)
    
    print(f"\n  Dataset da testare: {len(all_datasets)}")
    for name, path in all_datasets.items():
        exists = "✓" if path.exists() else "✗"
        print(f"    {exists} {name}: {path.name}")
    
    # Testa ogni dataset
    all_results = []
    
    for dataset_name, dataset_path in all_datasets.items():
        output_subdir = OUTPUT_DIR / dataset_name
        result = test_single_dataset(dataset_name, dataset_path, output_subdir)
        
        if result is not None:
            all_results.append(result)
    
    # Se abbiamo risultati, crea confronti
    if len(all_results) == 0:
        print(f"\n  ✗ Nessun dataset testato con successo")
        return
    
    print(f"\n{'='*80}")
    print(f"  TEST COMPLETATI: {len(all_results)}/{len(all_datasets)} dataset")
    print(f"{'='*80}")
    
    # Crea plot comparativi
    create_comparison_plots(all_results, OUTPUT_DIR)
    
    # Crea tabella riassuntiva
    summary_df = create_summary_table(all_results, OUTPUT_DIR)
    
    # Report finale
    print(f"\n{'='*80}")
    print("  ANALISI COMPLETA")
    print(f"{'='*80}")
    print(f"\n  📁 Directory output: {OUTPUT_DIR}")
    print(f"\n  📊 BEST DATASET:")
    best = summary_df.iloc[0]
    print(f"     🏆 {best['Dataset']}")
    print(f"        • N pazienti: {best['N_Patients']}")
    print(f"        • R² = {best['R2']:.3f}")
    print(f"        • MAE = {best['MAE']:.2f}%")
    print(f"        • RMSE = {best['RMSE']:.2f}%")
    
    print(f"\n  📂 Sottodirectory per dataset:")
    for result in all_results:
        print(f"     • {result['dataset_name']}/")
    
    print(f"\n  📈 Plot comparativi:")
    print(f"     • comparison_metrics_barplot.png")
    print(f"     • comparison_r2_vs_size.png")
    print(f"     • comparison_feature_importance_heatmap.png")
    print(f"     • comparison_error_distributions.png")
    
    print(f"\n  📄 Tabella riassuntiva:")
    print(f"     • overall_summary.csv")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
