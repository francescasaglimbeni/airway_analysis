import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter threading issues
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

BASE_DIR = Path(r"/content/airway_analysis/validation_pipeline/OSIC_metrics_validation")

# Dataset da unified_prediction/
UNIFIED_DIR = BASE_DIR / "unified_prediction"
DATASETS = {
    'strict': UNIFIED_DIR / "dataset_strict.csv",
    'balanced': UNIFIED_DIR / "dataset_balanced.csv",
    'all': UNIFIED_DIR / "dataset_all.csv",
    'traditional_only': UNIFIED_DIR / "dataset_traditional_only.csv",
    'both_targets': UNIFIED_DIR / "dataset_both_targets.csv",
    # Note: decline_only NON incluso perché non ha FVC_percent_week52
}

# Output
OUTPUT_DIR = Path(
    r"airway_analysis/validation_pipeline"
    r"/validation_test_models/week52pred_dataset_comparison"
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
# SEED = 42
SEED = 42
EPOCHS_MAX = 500
PATIENCE = 100
VAL_FRACTION = 0.20
N_INNER_SPLITS = 10

# Best configs dalla grid search (balanced)
BEST_CONFIG = {
    'hidden1': 16,
    'hidden2': 8,
    'dropout': 0.2,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'epochs_max': EPOCHS_MAX,
    'patience': PATIENCE,
    'val_fraction': VAL_FRACTION,
    'n_inner_splits': N_INNER_SPLITS,
    'seed': SEED,
    # Ridge best
    'ridge_alpha': 5.0,
    # Lasso best
    'lasso_alpha': 0.5,
    # RF best
    'rf_n_estimators': 100,
    'rf_max_depth': 2,
    'rf_min_samples_split': 5,
    'rf_min_samples_leaf': 2,
    # XGBoost best
    'xgb_n_estimators': 100,
    'xgb_max_depth': 2,
    'xgb_learning_rate': 0.1,
    'xgb_reg_alpha': 1.0,
    'xgb_reg_lambda': 1.0,
    # Ensemble best
    'ensemble_ridge_weight': 0.7,
    'ensemble_rf_weight': 0.3,
}

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
        '''results_df, all_importances, fold_curves = run_loocv(
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
                'seed': SEED,
                # XGBoost parameters (default conservative config)
                'xgb_n_estimators': 100,
                'xgb_max_depth': 3,
                'xgb_learning_rate': 0.05,
                'xgb_reg_alpha': 2.0,
                'xgb_reg_lambda': 2.0,
            }
        )'''
        
        results_df, all_importances, fold_curves = run_loocv(
            df=df_clean,
            features=FEATURES,
            target=TARGET,
            device=DEVICE,
            config=BEST_CONFIG
        )
                
        # 3. Calcola metriche
        print(f"\n  [3/4] Calcolo metriche aggregate...")
        summary_df = compute_aggregate_metrics(results_df)
        
        # Estrai metriche MLP
        '''mlp_row = summary_df[summary_df['Model'] == 'MLP (multi-feature)']
        r2 = float(mlp_row['R²'].values[0])
        mae = float(mlp_row['MAE'].values[0])
        rmse = float(mlp_row['RMSE'].values[0])
        
        print(f"        ✓ R² = {r2:.3f}, MAE = {mae:.2f}, RMSE = {rmse:.2f}")
        '''
        print("        ✓ Metriche per modello:")
        for _, row in summary_df.iterrows():
            print(f"           {row['Model']}: R²={float(row['R²']):.3f}, MAE={float(row['MAE']):.2f}, RMSE={float(row['RMSE']):.2f}")

        # Prendi il modello migliore per R² come riferimento del dataset
        best_model_row = summary_df.loc[summary_df['R²'].astype(float).idxmax()]
        r2 = float(best_model_row['R²'])
        mae = float(best_model_row['MAE'])
        rmse = float(best_model_row['RMSE'])
        best_model_name = best_model_row['Model']
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
        
        '''return {
            'dataset_name': dataset_name,
            'n_patients': n_patients,
            'R2': r2,
            'MAE': mae,
            'RMSE': rmse,
            'predictions_df': results_df,
            'summary_df': summary_df,
            'importances_df': importance_df
        }'''
        return {
            'dataset_name': dataset_name,
            'n_patients': n_patients,
            'R2': r2,
            'MAE': mae,
            'RMSE': rmse,
            'best_model': best_model_name,
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
    try:
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
            y_pos = bar.get_height() + 0.02 if bar.get_height() >= 0 else bar.get_height() - 0.2
            ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                    f'{val:.3f}\n(n={n})', ha='center', va='bottom' if bar.get_height() >= 0 else 'top', fontsize=9)
        
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
    except Exception as e:
        plt.close()
        print(f"  ⚠ Errore durante creazione barplot: {e}")
    
    # -------------------------------------------------------------------------
    # 2. Scatter plot: R² vs n_patients
    # -------------------------------------------------------------------------
    try:
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
    except Exception as e:
        plt.close()
        print(f"  ⚠ Errore durante creazione scatter plot: {e}")
    
    # -------------------------------------------------------------------------
    # 3. Feature importance comparison (heatmap)
    # -------------------------------------------------------------------------
    try:
        # Verifica prima se TUTTI i dataset hanno importances valide
        valid_importances = []
        valid_names = []
        
        for result in all_results:
            imp_df = result['importances_df']
            
            if imp_df.empty:
                continue
            
            # Prova diverse strutture possibili
            if 'Feature' in imp_df.columns and 'Importance_Std' in imp_df.columns:
                # Struttura standard
                avg_importance = imp_df.groupby('Feature')['Importance_Std'].mean()
                avg_importance = avg_importance.reindex(FEATURES, fill_value=0)
                valid_importances.append(avg_importance.values)
                valid_names.append(result['dataset_name'])
            elif 'Feature' in imp_df.columns and 'Importance' in imp_df.columns:
                # Usa 'Importance' invece di 'Importance_Std'
                avg_importance = imp_df.groupby('Feature')['Importance'].mean()
                avg_importance = avg_importance.reindex(FEATURES, fill_value=0)
                valid_importances.append(avg_importance.values)
                valid_names.append(result['dataset_name'])
            elif len(imp_df.columns) >= 2:
                # Prova con le prime due colonne
                col_feature = imp_df.columns[0]
                col_importance = imp_df.columns[1]
                avg_importance = imp_df.groupby(col_feature)[col_importance].mean()
                avg_importance = avg_importance.reindex(FEATURES, fill_value=0)
                valid_importances.append(avg_importance.values)
                valid_names.append(result['dataset_name'])
        
        if len(valid_importances) >= 2:  # Almeno 2 dataset per fare un confronto
            fig, ax = plt.subplots(figsize=(12, 8))
            
            importance_matrix = np.array(valid_importances)
            
            # Crea heatmap
            sns.heatmap(importance_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                        xticklabels=[f.replace('_', ' ') for f in FEATURES],
                        yticklabels=valid_names,
                        cbar_kws={'label': 'Feature Importance'}, ax=ax)
            
            ax.set_title('Confronto Feature Importance tra Dataset', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(output_dir / 'comparison_feature_importance_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Salvato: comparison_feature_importance_heatmap.png")
        else:
            plt.close()
            print(f"  ⚠ Skip: comparison_feature_importance_heatmap.png (dati insufficienti: {len(valid_importances)} dataset)")
    except Exception as e:
        plt.close()
        print(f"  ⚠ Errore durante creazione heatmap: {e}")
        import traceback
        traceback.print_exc()
    
    # -------------------------------------------------------------------------
    # 4. Box-plot degli errori per dataset
    # -------------------------------------------------------------------------
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        all_errors = []
        labels = []
        
        for result in all_results:
            df = result['predictions_df']
            
            # Estratto errori in base alla struttura disponibile
            if 'Error' in df.columns:
                # Se c'è già la colonna Error
                if 'Model' in df.columns:
                    # Filtra per MLP se esiste la colonna Model
                    mlp_data = df[df['Model'] == 'MLP (multi-feature)']
                    if len(mlp_data) > 0:
                        errors = mlp_data['Error'].values
                    else:
                        errors = df['Error'].values
                else:
                    # Usa tutte le righe
                    errors = df['Error'].values
            elif 'Actual' in df.columns and 'Predicted' in df.columns:
                # Calcola l'errore da Actual e Predicted
                if 'Model' in df.columns:
                    mlp_data = df[df['Model'] == 'MLP (multi-feature)']
                    if len(mlp_data) > 0:
                        errors = mlp_data['Predicted'].values - mlp_data['Actual'].values
                    else:
                        errors = df['Predicted'].values - df['Actual'].values
                else:
                    errors = df['Predicted'].values - df['Actual'].values
            else:
                # Salta questo dataset se non ci sono dati utilizzabili
                print(f"  ⚠ Colonne mancanti per {result['dataset_name']}: {list(df.columns)}")
                continue
            
            all_errors.append(errors)
            labels.append(f"{result['dataset_name']}\n(n={result['n_patients']})")
        
        if len(all_errors) > 0:
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
        else:
            plt.close()
            print(f"  ⚠ Skip: comparison_error_distributions.png (nessun dato valido)")
    except Exception as e:
        plt.close()
        print(f"  ⚠ Errore durante creazione boxplot: {e}")
        import traceback
        traceback.print_exc()


'''def create_summary_table(all_results, output_dir):
    """
    Crea tabella riassuntiva con tutte le metriche
    """
    print(f"\n{'='*80}")
    print("  CREAZIONE TABELLA RIASSUNTIVA")
    print(f"{'='*80}")
    
    try:
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
    except Exception as e:
        print(f"  ✗ Errore durante creazione tabella: {e}")
        # Ritorna DataFrame vuoto in caso di errore
        return pd.DataFrame({'Dataset': [r['dataset_name'] for r in all_results],
                           'R2': [r['R2'] for r in all_results]})
'''

def create_summary_table(all_results, output_dir):
    print(f"\n{'='*80}")
    print("  CREAZIONE TABELLA RIASSUNTIVA")
    print(f"{'='*80}")

    try:
        # Tabella per modello x dataset
        rows = []
        for result in all_results:
            for _, row in result['summary_df'].iterrows():
                rows.append({
                    'Dataset': result['dataset_name'],
                    'N_Patients': result['n_patients'],
                    'Model': row['Model'],
                    'R2': float(row['R²']),
                    'MAE': float(row['MAE']),
                    'RMSE': float(row['RMSE']),
                })

        df_long = pd.DataFrame(rows)
        df_long.to_csv(output_dir / 'overall_summary_all_models.csv', index=False)

        # Pivot: righe=dataset, colonne=modello
        pivot = df_long.pivot_table(
            index=['Dataset', 'N_Patients'],
            columns='Model',
            values='R2'
        ).reset_index()
        pivot.to_csv(output_dir / 'overall_summary_pivot_r2.csv', index=False)

        # Tabella best model per dataset (come prima ma corretta)
        summary_data = []
        for result in all_results:
            summary_data.append({
                'Dataset': result['dataset_name'],
                'N_Patients': result['n_patients'],
                'Best_Model': result['best_model'],
                'Best_R2': result['R2'],
                'Best_MAE': result['MAE'],
                'Best_RMSE': result['RMSE'],
            })

        df_best = pd.DataFrame(summary_data).sort_values('Best_R2', ascending=False)
        df_best.insert(0, 'Rank', range(1, len(df_best) + 1))
        df_best.to_csv(output_dir / 'overall_summary.csv', index=False)

        print(f"\n  RANKING DATASET (per best model R²):")
        print(df_best.to_string(index=False))

        print(f"\n  PIVOT R² (tutti i modelli):")
        print(pivot.to_string(index=False))

        return df_best, df_long

    except Exception as e:
        print(f"  ✗ Errore: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame()

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("  TEST PREDIZIONE WEEK52 SU DATASET UNIFIED V2")
    print("="*80)
    print(f"  Script sorgente: unified_fvc_prediction_v2.py")
    print(f"  Target: {TARGET}")
    print(f"  Features: {len(FEATURES)}")
    print(f"  Device: {DEVICE}")
    print(f"  Output: {OUTPUT_DIR}")
    print("="*80)
    
    # Crea directory output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n  Dataset da testare: {len(DATASETS)}")
    for name, path in DATASETS.items():
        exists = "✓" if path.exists() else "✗"
        print(f"    {exists} {name}: {path.name}")
    
    # Testa ogni dataset
    all_results = []
    
    for dataset_name, dataset_path in DATASETS.items():
        output_subdir = OUTPUT_DIR / dataset_name
        result = test_single_dataset(dataset_name, dataset_path, output_subdir)
        
        if result is not None:
            all_results.append(result)
    
    # Se abbiamo risultati, crea confronti
    if len(all_results) == 0:
        print(f"\n  ✗ Nessun dataset testato con successo")
        return
    
    print(f"\n{'='*80}")
    print(f"  TEST COMPLETATI: {len(all_results)}/{len(DATASETS)} dataset")
    print(f"{'='*80}")
    
    # Crea tabella riassuntiva (unico output comparativo)
    # summary_df = create_summary_table(all_results, OUTPUT_DIR)
    summary_df, summary_long = create_summary_table(all_results, OUTPUT_DIR)
    
    # Report finale
    print(f"\n{'='*80}")
    print("  ANALISI COMPLETA")
    print(f"{'='*80}")
    print(f"\n  📁 Directory output: {OUTPUT_DIR}")
    
    # Mostra best dataset solo se la tabella non è vuota
    if not summary_df.empty and len(summary_df) > 0:
        try:
            print(f"\n  📊 BEST DATASET:")
            best = summary_df.iloc[0]
            print(f"     🏆 {best['Dataset']}")
            if 'N_Patients' in best:
                print(f"        • N pazienti: {int(best['N_Patients'])}")
            print(f"        • Best Model: {best['Best_Model']}")
            print(f"        • R² = {best['Best_R2']:.3f}")
            if 'Best_MAE' in best:
                print(f"        • MAE = {best['Best_MAE']:.2f}%")
            if 'Best_RMSE' in best:
                print(f"        • RMSE = {best['Best_RMSE']:.2f}%")
        except Exception as e:
            print(f"  ⚠ Impossibile mostrare best dataset: {e}")
    
    print(f"\n  📂 Sottodirectory per dataset:")
    for result in all_results:
        print(f"     • {result['dataset_name']}/")
    
    print(f"\n  📄 Tabella riassuntiva: overall_summary.csv")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
