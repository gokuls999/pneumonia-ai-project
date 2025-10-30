import torch
import cv2
import numpy as np
from transformers import AutoTokenizer
import timm
import segmentation_models_pytorch as smp
import shap
import captum
import pgmpy
import optuna
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())  # Should be False since no GPU
print("All key imports successful!")