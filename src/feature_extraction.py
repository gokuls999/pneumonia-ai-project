import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Optional
from transformers import AutoModel
import timm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import PreprocessedMultiModalDataset
from torchvision import transforms

class FeatureExtractor(nn.Module):
    """
    Feature extraction module for multi-modal data.
    Extracts embeddings from text (Clinical BERT) and images (EfficientNet B4 for CXR, DenseNet 121 for CT).
    Applies global average pooling for projection.

    Args:
        text_model_name (str): Hugging Face model for text (e.g., 'emilyalsentzer/Bio_ClinicalBERT').
        cxr_model_name (str): TIMM model for CXR (e.g., 'efficientnet_b4').
        ct_model_name (str): TIMM model for CT (e.g., 'densenet121').
        pooling (str): Pooling type ('global_avg' default).
        device (str): Device ('cuda' or 'cpu').
    """
    def __init__(self, text_model_name: str = 'emilyalsentzer/Bio_ClinicalBERT', 
                 cxr_model_name: str = 'efficientnet_b4', ct_model_name: str = 'densenet121',
                 pooling: str = 'global_avg', device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.pooling = pooling
        
        # Text extractor (Clinical BERT)
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.text_model.to(device)
        self.text_model.eval()
        
        # Image extractors
        self.cxr_model = timm.create_model(cxr_model_name, pretrained=True, num_classes=0, global_pool='')
        self.cxr_model.to(device)
        self.cxr_model.eval()
        
        self.ct_model = timm.create_model(ct_model_name, pretrained=True, num_classes=0, global_pool='')
        self.ct_model.to(device)
        self.ct_model.eval()
        
        # Pooling layer
        if pooling == 'global_avg':
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            raise ValueError("Pooling must be 'global_avg'")

    def extract_text_features(self, tokenized_texts: dict) -> torch.Tensor:
        """Extract features from tokenized text using Clinical BERT."""
        with torch.no_grad():
            # Squeeze extra dim if present (e.g., [batch, 1, seq] -> [batch, seq])
            for k in tokenized_texts.keys():
                if tokenized_texts[k].dim() == 3:
                    tokenized_texts[k] = tokenized_texts[k].squeeze(1)
            outputs = self.text_model(**{k: v.to(self.device) for k, v in tokenized_texts.items()})
            features = outputs.last_hidden_state.mean(dim=1)  # Mean pooling over tokens
        return features

    def extract_image_features(self, image: torch.Tensor, is_ct: bool = False) -> torch.Tensor:
        """Extract features from image using appropriate model."""
        with torch.no_grad():
            if is_ct:
                model = self.ct_model
            else:
                model = self.cxr_model
            features = model(image.to(self.device))
            if self.pooling == 'global_avg':
                features = self.pool(features).flatten(1)  # Global average pooling
        return features

    def forward(self, batch: tuple) -> tuple:
        """
        Extract features for a batch.

        Args:
            batch: (tokenized_texts, cxr_images, ct_images, severities, classes) from dataset.

        Returns:
            (text_features, cxr_features, ct_features)
        """
        tokenized_texts, cxr_images, ct_images, severities, classes = batch
        
        text_features = self.extract_text_features(tokenized_texts)
        cxr_features = self.extract_image_features(cxr_images, is_ct=False)
        ct_features = self.extract_image_features(ct_images, is_ct=True)
        
        return text_features, cxr_features, ct_features

# Example usage with Phase 2 dataset
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = PreprocessedMultiModalDataset(data_dir='data/raw', image_transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Extractor
    extractor = FeatureExtractor(device='cpu')
    
    for batch in dataloader:
        text_feats, cxr_feats, ct_feats = extractor(batch)
        print("Feature extraction successful!")
        print(f"Text features shape: {text_feats.shape}")
        print(f"CXR features shape: {cxr_feats.shape}")
        print(f"CT features shape: {ct_feats.shape}")
        break