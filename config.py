import torch
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"

MODEL_PATH = "models/pcos_model.pth"
OUTPUT_DIR = "outputs/"

IMAGE_SIZE = 224
BATCH_SIZE = 8

EPOCHS = 10
LEARNING_RATE = 0.0001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASSES = ["normal", "pcos"]