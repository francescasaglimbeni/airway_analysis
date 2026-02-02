"""
Funzioni per I/O e salvataggio risultati
"""
import pandas as pd
import json
from pathlib import Path


def save_results(results_df, importances, fold_curves, output_dir, features):
    """
    Salva tutti i risultati in file.
    
    Args:
        results_df: DataFrame con risultati LOOCV
        importances: Lista di dizionari con feature importance
        fold_curves: Lista di dizionari con curve di training
        output_dir: Directory di output
        features: Lista delle feature
    """
    # Salva predizioni LOOCV
    results_df.to_csv(output_dir / 'loocv_predictions.csv', index=False)
    print(f"  ✓ loocv_predictions.csv salvato ({len(results_df)} pazienti)")
    
    # Salva feature importances
    if importances:
        importance_df = pd.DataFrame(importances)
        importance_df.to_csv(output_dir / 'feature_importances.csv', index=False)
        print(f"  ✓ feature_importances.csv salvato ({len(importances)} fold)")
    
    # Salva training curves (solo metadati)
    if fold_curves:
        curves_metadata = []
        for curve in fold_curves:
            curves_metadata.append({
                'fold': curve['fold'],
                'patient': curve['patient'],
                'train_losses_length': len(curve['train_losses']),
                'val_losses_length': len(curve['val_losses']),
                'final_train_loss': curve['train_losses'][-1] if curve['train_losses'] else None,
                'final_val_loss': curve['val_losses'][-1] if curve['val_losses'] else None,
            })
        
        curves_df = pd.DataFrame(curves_metadata)
        curves_df.to_csv(output_dir / 'training_curves_metadata.csv', index=False)
        print(f"  ✓ training_curves_metadata.csv salvato ({len(fold_curves)} curve)")
    
    # Salva configurazione
    config_summary = {
        'n_patients': len(results_df),
        'n_features': len(features),
        'features': features,
        'target': 'FVC_percent_week52',
    }
    
    with open(output_dir / 'config_summary.json', 'w') as f:
        json.dump(config_summary, f, indent=2)
    
    print(f"  ✓ config_summary.json salvato")