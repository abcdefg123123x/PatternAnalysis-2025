import os
import random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Hyperparameters
BATCH_SIZE = 32
WORKERS = 4
TRAIN_SPLIT = 0.7
TEST_SPLIT = 0.2
VAL_SPLIT = 0.1

class ISICDataset(Dataset):
    """Dataset for ISIC 2020 Kaggle Challenge (triplet sampling)"""
    def __init__(self, image_root, samples, transform=None):
        self.image_root = image_root
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        anchor = self.samples[idx]
        positive = random.choice(self.samples)
        while positive['target'] != anchor['target']:
            positive = random.choice(self.samples)
        negative = random.choice(self.samples)
        while negative ['target'] == anchor['target']:
            negative = random.choice(self.samples)

        def load_img(sample):
            img = Image.open(os.path.join(self.image_root,
                                sample['isic_id' + ".jpg"])).convert('RGB')

        return (load_img(anchor), load_img(positive), load_img(negative),
                anchor['target'])