import os
import torch
from dotenv import load_dotenv
load_dotenv()

MODEL_NAME = os.getenv('MODEL_NAME', 'efficientnet_b0')
MODEL_PATH = os.getenv('MODEL_PATH', 'efficientnet_finetuned_final.pth')

NUM_CLASSES = 4
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]