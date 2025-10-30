import os
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from PIL import Image
from typing import Tuple, List, Optional

class MultiModalPneumoniaDataset(Dataset):
    """
    Custom PyTorch Dataset for loading multi-modal pneumonia data: text (medical records), CXR images, CT images.
    Assumes data is paired by patient ID.

    Args:
        data_dir (str): Path to the raw data directory (e.g., 'data/raw').
        transform (Optional[callable]): Optional transform for images (e.g., from torchvision.transforms).
        text_column (str): Column name for medical text in CSV.
        label_column_severity (str): Column for severity score label.
        label_column_class (str): Column for bacterial/viral class label.

    Returns:
        tuple: (text, cxr_image, ct_image, severity_label, class_label)
    """
    def __init__(self, data_dir: str, transform: Optional[callable] = None, text_column: str = 'notes', 
                 label_column_severity: str = 'severity', label_column_class: str = 'pneumonia_type'):
        self.data_dir = data_dir
        self.transform = transform
        self.text_column = text_column
        self.label_column_severity = label_column_severity
        self.label_column_class = label_column_class
        
        # Load metadata CSV (assume 'mimic_cxr_metadata.csv' with columns: patient_id, text, cxr_path, ct_path, labels)
        self.metadata = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
        self.patient_ids = self.metadata['patient_id'].tolist()

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, torch.Tensor, float, int]:
        patient_id = self.patient_ids[idx]
        row = self.metadata[self.metadata['patient_id'] == patient_id].iloc[0]
        
        # Load text
        text = row[self.text_column]
        
        # Load CXR image
        cxr_path = os.path.join(self.data_dir, row['cxr_path'])
        cxr_image = Image.open(cxr_path).convert('RGB')
        if self.transform:
            cxr_image = self.transform(cxr_image)
        
        # Load CT image
        ct_path = os.path.join(self.data_dir, row['ct_path'])
        ct_image = Image.open(ct_path).convert('L')  # Grayscale for CT
        if self.transform:
            ct_image = self.transform(ct_image)
        
        # Labels
        severity_label = row[self.label_column_severity]  # e.g., float score 0-10
        class_label = row[self.label_column_class]  # e.g., 0 for bacterial, 1 for viral
        
        return text, cxr_image, ct_image, severity_label, class_label

# Example usage (test in a notebook)
if __name__ == "__main__":
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    dataset = MultiModalPneumoniaDataset(data_dir='data/raw', transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for batch in dataloader:
        texts, cxrs, cts, severities, classes = batch
        print("Sample batch loaded successfully!")
        break