import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = os.path.expanduser("./data")
BATCH_SIZE = 8
NUM_WORKERS = 4
MODEL_PATH = "./models"
HIDDEN_DIM = 768
SCALE = 2 << 11
SEQ_LEN = 197
