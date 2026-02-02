import numpy as np
from sklearn.linear_model import LinearRegression


def run_linear_baselines(X_train, y_train, X_test, y_test, feature_names):
    """
    Per ogni feature, addestra un Linear Regression sul train set
    e predice sul test set.
    
    Args:
        X_train, y_train: Dati di training
        X_test, y_test: Dati di test
        feature_names: Lista dei nomi delle feature
    
    Returns:
        baselines: dict {feature_name: predicted_value}
    """
    baselines = {}
    for i, fname in enumerate(feature_names):
        x_tr = X_train[:, i].reshape(-1, 1)
        x_te = X_test[:, i].reshape(-1, 1)
        
        # Skip se la feature ha varianza zero nel train
        if np.std(x_tr) < 1e-10:
            baselines[fname] = np.nan
            continue
        
        lr = LinearRegression()
        lr.fit(x_tr, y_train)
        baselines[fname] = lr.predict(x_te)[0]
    
    return baselines