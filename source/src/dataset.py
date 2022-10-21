import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A
import cv2
import os


class ReagingDataset(Dataset):
    def __init__(self, root: str):
        super(ReagingDataset, self).__init__()
        self.root = root
        self.dirs = os.listdir(self.root)

    def __len__(self):
        num_images = 0
        for dir in self.dirs:
            num_images += len(os.listdir(os.path.join(self.root, dir)))
        return num_images
    
    def __getitem__(self, ix: int):
        pass


