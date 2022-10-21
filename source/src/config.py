import torch
import albumentations as A

DEVICE = torch.device("cuda" if torch.has_cuda else "mps" if torch.has_mps else "cpu")
LEARNING_RATE = 2e-4
DATA_DIR = "Users/aneeshaparajit/Desktop/data"
BATCH_SIZE = 16
IMAGE_SIZE = 128
IMG_CHANNELS = 3
NUM_EPOCHS = 500
L1_LAMBDA = 100

# UPDATE BEFORE RUNNING
DISC_CHECKPOINT = "data/to/disc/weights"
GEN_CHECKPOINT = "data/to/gen/weights"


