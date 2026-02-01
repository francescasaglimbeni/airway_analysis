"""
Definizione del modello MLP per predizione FVC%
"""
import torch.nn as nn


class FVC_MLP(nn.Module):
    """
    Architettura conservativa per predizione FVC% con poco dato.
    
    Input(n) → Linear(16) → ReLU → Dropout → Linear(8) → ReLU → Linear(1)
    
    Il bottleneck (8 neuroni) forza il modello a comprimere
    l'informazione delle feature in una rappresentazione densa
    prima della predizione finale.
    """
    def __init__(self, input_dim, hidden1=16, hidden2=8, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)