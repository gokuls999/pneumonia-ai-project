import os
import re
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import cv2
import numpy as np
from typing import Optional, Callable
from transformers import AutoTokenizer
from torchvision import transforms
import pandas as pd

class PreprocessedMultiModalDataset(Dataset):
    """
    Enhanced Dataset with pre-processing for multi-modal pneumonia data.
    Applies image pre-processing and text cleaning/tokenization.

    Args:
        data_dir (str): Path to raw data directory.
        tokenizer (Optional[Callable]): Tokenizer for text (e.g., Clinical BERT).
        image_transform (Optional[Callable]): Additional image transforms.
        apply_augmentation (bool): Whether to apply data augmentation.
    """
    def __init__(self, data_dir: str, tokenizer: Optional[Callable] = None, image_transform: Optional[Callable] = None,
                 apply_augmentation: bool = False):
        self.data_dir = data_dir
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        self.image_transform = image_transform
        self.apply_augmentation = apply_augmentation
        
        # Load metadata
        self.metadata = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
        self.patient_ids = self.metadata['patient_id'].tolist()

    def preprocess_image(self, image_path: str, is_ct: bool = False) -> np.ndarray:
        """
        Pre-process image: noise reduction, CLAHE, normalization, augmentation, resizing.

        Args:
            image_path (str): Path to image file.
            is_ct (bool): True for CT (grayscale), False for CXR (RGB).

        Returns:
            np.ndarray: Pre-processed image array (224x224x3).
        """
        # Load image
        full_path = os.path.join(self.data_dir, image_path)
        image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE if is_ct else cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not load image from {full_path}")

        # Handle channels correctly
        if is_ct:
            # Keep grayscale (1 channel)
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            # RGB for CXR (3 channels)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Step 1: Noise Reduction (Bilateral Filtering)
        if len(image.shape) == 3:
            for c in range(image.shape[2]):
                image[:,:,c] = cv2.bilateralFilter(image[:,:,c], 9, 75, 75)
        else:
            image = cv2.bilateralFilter(image, 9, 75, 75)

        # Step 2: Contrast Enhancement (CLAHE)
        if is_ct:
            # Grayscale CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            image = clahe.apply(image)
            # Repeat grayscale to 3 channels for CNN (R=G=B=grayscale)
            image = np.repeat(image[:,:,np.newaxis], 3, axis=2)
        else:
            # LAB for RGB
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            image = cv2.merge((cl, a, b))
            image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)

        # Step 3: Image Normalization (with epsilon for zero std)
        image = image.astype(np.float32)
        mean = np.mean(image)
        std = np.std(image)
        if std == 0:
            std = 1e-8  # Epsilon to avoid division by zero
        image = (image - mean) / std

        # Step 4: ROI Segmentation (stub for U-Net + EfficientNet; implement full model later)
        if False:  # Placeholder - load roi_model when available
            pass

        # Step 5: Data Augmentation (if enabled)
        if self.apply_augmentation:
            image = self.augment_image(image)

        # Step 6: Image Resizing (224x224)
        image = cv2.resize(image, (224, 224))

        return image

    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """Basic augmentation (rotation, flip)."""
        h, w = image.shape[:2]
        angle = np.random.uniform(-15, 15)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
        return image

    def preprocess_text(self, text: str) -> dict:
        """Clean and tokenize text for Clinical BERT."""
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^a-zA-Z0-9\s.,!?;:\-]', '', text)
        tokens = self.tokenizer.encode_plus(text, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': tokens['input_ids'], 'attention_mask': tokens['attention_mask']}

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> tuple:
        patient_id = self.patient_ids[idx]
        row = self.metadata[self.metadata['patient_id'] == patient_id].iloc[0]
        
        # Pre-process text
        text = row['notes']
        tokenized_text = self.preprocess_text(text)
        
        # Pre-process CXR
        cxr_path = row['cxr_path']
        cxr_image = self.preprocess_image(cxr_path, is_ct=False)
        if self.image_transform:
            cxr_image = self.image_transform(cxr_image)
        
        # Pre-process CT
        ct_path = row['ct_path']
        ct_image = self.preprocess_image(ct_path, is_ct=True)
        if self.image_transform:
            ct_image = self.image_transform(ct_image)
        
        # Labels
        severity_label = row['severity']
        class_label = row['pneumonia_type']
        
        return tokenized_text, cxr_image, ct_image, severity_label, class_label

# Example usage
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = PreprocessedMultiModalDataset(data_dir='data/raw', image_transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    for batch in dataloader:
        tokenized_texts, cxrs, cts, severities, classes = batch
        print("Pre-processed batch loaded successfully!")
        break