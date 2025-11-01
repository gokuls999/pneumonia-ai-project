import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Optional
import optuna
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import PreprocessedMultiModalDataset
from torchvision import transforms

class ProbabilisticOutput(nn.Module):
    """
    Probabilistic output layer using Bayesian Network for uncertainty.
    Integrates with pgmpy for Bayesian inference on multi-task predictions.
    """
    def __init__(self, severity_dim: int = 1, class_dim: int = 2, device: str = 'cpu'):
        super().__init__()
        self.device = device
        
        # Bayesian Network (simple dependency: severity influences class)
        self.bn_model = DiscreteBayesianNetwork([('severity', 'class')])
        
        # Define CPDs with correct shapes
        cpd_severity = TabularCPD('severity', 3, [[0.3], [0.4], [0.3]])  # Shape (3, 1) for 3 states
        cpd_class = TabularCPD('class', 2, [[0.7, 0.4, 0.2], [0.3, 0.6, 0.8]], 
                               evidence=['severity'], evidence_card=[3])  # Shape (2, 3) for 2 states, 3 evidence
        
        self.bn_model.add_cpds(cpd_severity, cpd_class)
        self.bn_model.check_model()
        self.infer = VariableElimination(self.bn_model)
        
        # Softmax for class probabilities
        self.softmax = nn.Softmax(dim=1)

    def forward(self, severity_logits: torch.Tensor, class_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Forward pass with Bayesian inference for probabilistic outputs.

        Args:
            severity_logits: [batch, 1] from severity head.
            class_logits: [batch, 2] from class head.

        Returns:
            (severity_probs, class_probs, bayesian_evidence): Probabilities and Bayesian evidence dict.
        """
        severity_probs = torch.sigmoid(severity_logits)  # [batch, 1]
        class_probs = self.softmax(class_logits)  # [batch, 2]
        
        # Bayesian inference (dummy evidence; in practice, add prior evidence from data)
        bayesian_evidence = {'evidence': {}, 'probabilities': {}}
        for i in range(severity_probs.shape[0]):
            evidence = {'severity': int(severity_probs[i].item() * 3)}  # Discretize to 0-2
            batch_evidence = self.infer.query(variables=['class'], evidence=evidence)
            bayesian_evidence['evidence'][i] = evidence
            bayesian_evidence['probabilities'][i] = batch_evidence
        
        return severity_probs, class_probs, bayesian_evidence

class DeepLearningBackbone(nn.Module):
    """
    Complete model with multi-task learning and probabilistic output.
    """
    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, num_layers: int = 6, num_heads: int = 8,
                 dropout_rate: float = 0.1, l2_lambda: float = 1e-4, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.l2_lambda = l2_lambda
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Transformer backbone
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
                                                   dropout=dropout_rate, activation='relu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Leaky ReLU and dropout
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Multi-task heads (shared backbone)
        self.severity_head = nn.Linear(hidden_dim, 1)
        self.class_head = nn.Linear(hidden_dim, 2)
        
        # Probabilistic output
        self.probabilistic_output = ProbabilisticOutput(device=device)

    def forward(self, fused_feats: torch.Tensor, use_mixed_precision: bool = False) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Forward pass with probabilistic output.

        Args:
            fused_feats: [batch, fusion_dim] from fusion.
            use_mixed_precision: Enable FP16.

        Returns:
            (severity_probs, class_probs, bayesian_evidence)
        """
        if use_mixed_precision:
            with torch.autocast(device_type='cpu'):
                fused_feats = fused_feats.half()
        else:
            fused_feats = fused_feats.float()
            
        fused_feats = fused_feats.float()    
        
        # Project input
        x = self.input_proj(fused_feats)  # [batch, hidden_dim]
        x = x.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Transformer backbone
        x = self.transformer(x)  # [batch, 1, hidden_dim]
        x = x.squeeze(1)  # [batch, hidden_dim]
        
        # Leaky ReLU and dropout
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        # Multi-task logits
        severity_logits = self.severity_head(x)  # [batch, 1]
        class_logits = self.class_head(x)  # [batch, 2]
        
        if self.training:
            severity_probs = torch.sigmoid(severity_logits)
            class_probs = torch.softmax(class_logits, dim=1)
            bayesian_evidence = None
        else:
            severity_probs, class_probs, bayesian_evidence = self.probabilistic_output(severity_logits, class_logits)
            
        return severity_probs, class_probs, bayesian_evidence        

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 10, device: str = 'cpu'):
    """
    Training pipeline with multi-task loss and mixed precision.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=model.l2_lambda)
    criterion_severity = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss()
    
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            tokenized_texts, cxrs, cts, severities, classes = batch
            # Dummy fused feats (replace with real fusion in full pipeline)
            fused_feats = torch.randn(severities.shape[0], 768, device=device)
            
            optimizer.zero_grad()
            severity_probs, class_probs, _ = model(fused_feats, use_mixed_precision=True)
            
            loss_severity = criterion_severity(severity_probs.squeeze(), severities.float())
            loss_class = criterion_class(class_probs, classes.long())
            loss = loss_severity + loss_class
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                tokenized_texts, cxrs, cts, severities, classes = batch
                fused_feats = torch.randn(severities.shape[0], 768, device=device)
                severity_probs, class_probs, _ = model(fused_feats, use_mixed_precision=False)
                
                loss_severity = criterion_severity(severity_probs.squeeze(), severities.float())
                loss_class = criterion_class(class_probs, classes.long())
                loss = loss_severity + loss_class
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

# Example usage
if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    
    dataset = PreprocessedMultiModalDataset(data_dir='data/raw', image_transform=transform)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Complete model with probabilistic output
    model = DeepLearningBackbone(input_dim=768, device='cpu')
    
    # Test forward pass
    dummy_fused = torch.randn(2, 768)
    severity_probs, class_probs, evidence = model(dummy_fused, use_mixed_precision=False)
    print("Multi-task model with probabilistic output successful!")
    print(f"Severity probs shape: {severity_probs.shape}")
    print(f"Class probs shape: {class_probs.shape}")
    
    if evidence is not None:
        print(f"Bayesian evidence keys: {list(evidence['evidence'].keys())}")
    else:
        print("Bayesian evidence skipped (training mode).")    
    
    # Train (dummy; use real data in full run)
    train_model(model, train_loader, train_loader, epochs=1)