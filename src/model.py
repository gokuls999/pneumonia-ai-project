import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Optional
import optuna
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fusion import MultiModalFusion
from src.feature_extraction import FeatureExtractor
from src.preprocessing import PreprocessedMultiModalDataset
from torchvision import transforms

class DeepLearningBackbone(nn.Module):
    """
    Deep learning backbone for the multi-modal pneumonia system.
    Uses an efficient transformer encoder with Leaky ReLU, Dropout/L2 regularization, and mixed precision support.

    Args:
        input_dim (int): Dimension of fused input features (e.g., 768 from fusion).
        hidden_dim (int): Hidden dimension for transformer layers.
        num_layers (int): Number of transformer layers (default 6).
        num_heads (int): Number of attention heads (default 8).
        dropout_rate (float): Dropout rate (default 0.1).
        l2_lambda (float): L2 regularization lambda (default 1e-4).
        device (str): Device ('cuda' or 'cpu').
    """
    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, num_layers: int = 6, num_heads: int = 8,
                 dropout_rate: float = 0.1, l2_lambda: float = 1e-4, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.l2_lambda = l2_lambda
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Transformer backbone (efficient encoder)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
                                                   dropout=dropout_rate, activation='relu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Leaky ReLU activation
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Multi-task outputs (severity score as regression, classification as 2-class)
        self.severity_head = nn.Linear(hidden_dim, 1)
        self.class_head = nn.Linear(hidden_dim, 2)

    def forward(self, fused_feats: torch.Tensor, use_mixed_precision: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the backbone.

        Args:
            fused_feats: [batch, fusion_dim] from fusion layer.
            use_mixed_precision: If True, use FP16 (requires amp context).

        Returns:
            (severity_logits, class_logits): [batch, 1] and [batch, 2]
        """
        if use_mixed_precision:
            fused_feats = fused_feats.half()  # FP16
        
        # Project input
        x = self.input_proj(fused_feats)  # [batch, hidden_dim]
        
        # Add sequence dimension for transformer (treat as seq_len=1)
        x = x.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Transformer backbone
        x = self.transformer(x)  # [batch, 1, hidden_dim]
        x = x.squeeze(1)  # [batch, hidden_dim]
        
        # Leaky ReLU and dropout
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        # Multi-task outputs
        severity_logits = self.severity_head(x)  # [batch, 1]
        class_logits = self.class_head(x)  # [batch, 2] for bacterial/viral
        
        return severity_logits, class_logits

def tune_hyperparameters(trial: optuna.Trial, dataloader: DataLoader, device: str = 'cpu') -> float:
    """
    Optuna objective for hyperparameter tuning.

    Args:
        trial: Optuna trial object.
        dataloader: Validation DataLoader.
        device: Device to use.

    Returns:
        float: Validation loss.
    """
    # Suggest hyperparameters
    hidden_dim = trial.suggest_categorical('hidden_dim', [256, 512, 1024])
    num_layers = trial.suggest_int('num_layers', 4, 8)
    num_heads = trial.suggest_categorical('num_heads', [4, 8, 16])
    dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.2)
    l2_lambda = trial.suggest_float('l2_lambda', 1e-5, 1e-3)
    
    # Create model
    model = DeepLearningBackbone(input_dim=768, hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads,
                                 dropout_rate=dropout_rate, l2_lambda=l2_lambda, device=device)
    model.to(device)
    
    # Dummy training (replace with real in Phase 6)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=l2_lambda)
    criterion_severity = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss()
    
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        tokenized_texts, cxrs, cts, severities, classes = batch
        # Dummy fused feats (replace with real fusion in Phase 6)
        fused_feats = torch.randn(severities.shape[0], 768, device=device)
        
        optimizer.zero_grad()
        severity_logits, class_logits = model(fused_feats, use_mixed_precision=False)
        
        loss_severity = criterion_severity(severity_logits.squeeze(), severities.float())
        loss_class = criterion_class(class_logits, classes.long())
        loss = loss_severity + loss_class
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

# Example usage
if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    
    dataset = PreprocessedMultiModalDataset(data_dir='data/raw', image_transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Example backbone
    backbone = DeepLearningBackbone(input_dim=768, device='cpu')
    
    # Test forward pass (with dummy fused feats from fusion)
    dummy_fused = torch.randn(2, 768)
    severity_logits, class_logits = backbone(dummy_fused)
    print("Deep learning backbone successful!")
    print(f"Severity logits shape: {severity_logits.shape}")  # [2, 1]
    print(f"Class logits shape: {class_logits.shape}")  # [2, 2]

    # Example tuning (uncomment for real tuning in Phase 6)
    # study = optuna.create_study(direction='minimize')
    # study.optimize(lambda trial: tune_hyperparameters(trial, dataloader), n_trials=10)
    # print("Best hyperparameters:", study.best_params)