import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple
import timm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature_extraction import FeatureExtractor
from src.preprocessing import PreprocessedMultiModalDataset
from torchvision import transforms

class MultiModalFusion(nn.Module):
    """
    Multi-modal fusion layer using attention-based cross-modal transformers.
    Fuses text, CXR, and CT features with cross-attention, then projects via ViT as sequence processor (adapted for 1D features).
    """
    def __init__(self, text_dim: int = 768, cxr_dim: int = 1792, ct_dim: int = 1024, fusion_dim: int = 512,
                 num_heads: int = 8, num_layers: int = 2, device: str = 'cpu'):
        super().__init__()
        self.device = device
        
        # Linear projections to common dimension
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.cxr_proj = nn.Linear(cxr_dim, fusion_dim)
        self.ct_proj = nn.Linear(ct_dim, fusion_dim)
        
        # Cross-modal transformer (text as query, images as key/value)
        encoder_layer = nn.TransformerEncoderLayer(d_model=fusion_dim, nhead=num_heads, dim_feedforward=fusion_dim*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # ViT for final fusion (adapted for sequence input, bypassing image-specific parts)
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0, global_pool='avg')
        self.vit.to(device)
        self.vit.eval()
        self.vit.patch_embed = nn.Identity()  # Skip patch embedding
        self.vit.pos_embed = nn.Parameter(torch.zeros(1, 1, self.vit.embed_dim))  # Dummy position embedding for seq_len=1
        
        # Projection to ViT embed_dim (768)
        self.to_vit_embed = nn.Linear(fusion_dim, self.vit.embed_dim)
        
        # Final linear for fused features
        self.fusion_linear = nn.Linear(fusion_dim * 3, fusion_dim)

    def forward(self, text_feats: torch.Tensor, cxr_feats: torch.Tensor, ct_feats: torch.Tensor) -> torch.Tensor:
        """
        Fuse features using cross-attention and ViT.

        Args:
            text_feats: [batch, text_dim]
            cxr_feats: [batch, cxr_dim]
            ct_feats: [batch, ct_dim]

        Returns:
            [batch, fusion_dim]
        """
        batch_size = text_feats.shape[0]
        
        # Project to common dimension
        text_proj = self.text_proj(text_feats)  # [batch, fusion_dim]
        cxr_proj = self.cxr_proj(cxr_feats)  # [batch, fusion_dim]
        ct_proj = self.ct_proj(ct_feats)  # [batch, fusion_dim]
        
        # Reshape to [batch, seq_len=1, fusion_dim]
        text_seq = text_proj.unsqueeze(1)
        cxr_seq = cxr_proj.unsqueeze(1)
        ct_seq = ct_proj.unsqueeze(1)
        
        # Concat images as key/value
        image_kv = torch.cat([cxr_seq, ct_seq], dim=1)  # [batch, 2, fusion_dim]
        
        # Cross-attention: text queries image features
        cross_attended = self.transformer(text_seq)  # [batch, 1, fusion_dim]
        
        # Concat all
        fused = torch.cat([cross_attended.squeeze(1), cxr_proj, ct_proj], dim=1)  # [batch, fusion_dim*3]
        fused = self.fusion_linear(fused)  # [batch, fusion_dim]
        
        # Project to ViT embed_dim and treat as sequence of 1 patch [batch, 1, 768]
        vit_input = self.to_vit_embed(fused).unsqueeze(1)  # [batch, 1, 768]
        
        # Add position embedding
        vit_input = vit_input + self.vit.pos_embed
        
        # Pass to ViT transformer blocks directly (skip forward_features, use blocks)
        x = vit_input
        for block in self.vit.blocks:
            x = block(x)
        x = self.vit.norm(x)  # [batch, 1, 768]
        vit_features = x.mean(dim=1)  # Global average pooling over sequence [batch, 768]

        return vit_features

# Example usage
if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    
    dataset = PreprocessedMultiModalDataset(data_dir='data/raw', image_transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Feature extractor from Phase 3
    extractor = FeatureExtractor(device='cpu')
    
    # Fusion module
    fusion = MultiModalFusion(device='cpu')
    
    for batch in dataloader:
        text_feats, cxr_feats, ct_feats = extractor(batch)
        fused = fusion(text_feats, cxr_feats, ct_feats)
        print("Multi-modal fusion successful!")
        print(f"Fused features shape: {fused.shape}")
        break