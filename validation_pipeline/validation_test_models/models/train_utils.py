"""
Funzioni di training e valutazione per MLP
"""
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error

from .fvc_mlp import FVC_MLP


def train_mlp(X_train, y_train, X_val, y_val, device, config):
    """
    Trains MLP with early stopping on validation MAE.
    
    Args:
        X_train, y_train: Dati di training
        X_val, y_val: Dati di validazione
        device: torch.device ('cuda' o 'cpu')
        config: Dizionario con iperparametri
    
    Returns:
        model: Modello addestrato
        train_losses: Lista delle loss di training
        val_losses: Lista delle loss di validazione
    """
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    input_dim = X_train.shape[1]
    model = FVC_MLP(
        input_dim,
        hidden1=config.get('hidden1', 16),
        hidden2=config.get('hidden2', 8),
        dropout=config.get('dropout', 0.2)
    ).to(device)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get('learning_rate', 1e-3),
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    criterion = torch.nn.MSELoss()
    
    # DataLoader per train
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train).to(device),
        torch.FloatTensor(y_train).to(device)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(len(X_train), 32),  # Batch size più piccolo per stabilità
        shuffle=True
    )
    
    # Tensori val
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    
    train_losses, val_losses = [], []
    best_val_mae = float('inf')
    best_state = None
    patience_counter = 0
    epochs_max = config.get('epochs_max', 500)
    patience = config.get('patience', 50)
    
    for epoch in range(epochs_max):
        # --- TRAIN ---
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(X_batch)
        train_losses.append(epoch_loss / len(X_train))
        
        # --- VALIDATION ---
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_mse = criterion(val_pred, y_val_t).item()
            val_mae = mean_absolute_error(
                y_val.flatten(),
                val_pred.cpu().numpy().flatten()
            )
        val_losses.append(val_mse)
        
        # Early stopping sul MAE di val
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Ricarica il best model
    model.load_state_dict(best_state)
    model.to(device)
    
    return model, train_losses, val_losses


def predict_mlp(model, X, device):
    """
    Returns numpy array of predictions.
    
    Args:
        model: Modello MLP addestrato
        X: Dati di input
        device: torch.device
    
    Returns:
        predictions: Array numpy con predizioni
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        predictions = model(X_tensor).cpu().numpy().flatten()
    return predictions