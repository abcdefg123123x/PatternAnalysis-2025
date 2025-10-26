"""
dataset.py

Handles loading, preprocessing, and augmentation of the ISIC 2020 Kaggle
Challenge dataset. Supports triplet sampling for Siamese network training and
allows fraction-based selection of benign images.

Note:
    - benign_fraction: Default is 1.0 to use all benign images, and effectively
      all images are being used. This ensures full dataset utilisation but
      increases computational cost.
"""

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

# Standard transform for benign images
base_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# Apply strong transforms for malignant images to handle class imbalance
# AI used here: ChatGPT generated these stronger augmentations for malignant
malignant_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3,
                           hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1),
                            shear=10),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

class ISICDataset(Dataset):
    """
    PyTorch Dataset for the ISIC 2020 Kaggle Challenge.

    Implements triplet sampling:
        - anchor: a random image from the dataset
        - positive: a random image of the same class as the anchor
        - negative: a random image of a different class
    """
    def __init__(self, image_root, samples):
        """
        Intialise the dataset.

        Args:
            image_root (str): Path to folder containing the images.
            samples (list of dict): Metadata samples, each with 'isic_id' and
            'target'.
        """
        self.image_root = image_root
        self.samples = samples

    def __len__(self):
        """Returns the total number of samples in the dataset"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns a triplet (anchor, positive, negative) along with the label of
        the anchor.

        Args:
            idx (int): Index of the anchor sample.

        Returns:
            tuple: (anchor_img, positive_img, negative_img, label)
                - anchor_img (Tensor)
                - positive_img (Tensor)
                - negative_img (Tensor)
                - label (int)
        """
        anchor = self.samples[idx]

        # Select positive sample (same class)
        positive = random.choice(self.samples)
        while positive['target'] != anchor['target']:
            positive = random.choice(self.samples)

        # Select negative sample (different class)
        negative = random.choice(self.samples)
        while negative ['target'] == anchor['target']:
            negative = random.choice(self.samples)

        def load_img(sample):
            """
            Load and transform an image sample.

            Args:
                sample (dict): Metadata dictionary containing 'isic_id' and
                'target'.

            Returns:
                Tensor: Transformed image
            """
            img = Image.open(os.path.join(self.image_root,
                                          sample['isic_id'] + ".jpg")).convert(
                'RGB')

            # Apply stronger augmentations for malignant images
            if sample['target'] == 1:
                return malignant_transform(img)
            else:
                return base_transform(img)

        return (load_img(anchor), load_img(positive), load_img(negative),
                anchor['target'])

def split_data(csv_path, benign_fraction=1.0, seed=None):
    """
    Split the dataset into training, testing, and validation sets.

    Keeps all malignant images and allows fraction-based selection of benign
    images.

    Args:
        csv_path (str): Path to CSV file with dataset metadata.
        benign_fraction (float, 0-1]: Fraction of benign images to include.
        seed (int, optional): A seed for reproducibility.

    Returns:
        tuple: (train_samples, test_samples, val_samples) as a list of dicts.
    """
    if seed is not None:
        random.seed(seed)
        pd.np.random.seed(seed)

    df = pd.read_csv(csv_path)
    malignant = df[df['target'] == 1].copy()  # Keep all malignant images
    benign = df[
        df['target'] == 0].copy()  # Keep all benign initially

    # Keep the desired fraction of benign images
    n_benign = int(len(benign) * benign_fraction)
    benign = benign.sample(n=n_benign, random_state=seed)

    # Combine and shuffle
    df = pd.concat([malignant, benign]).sample(frac=1, random_state=seed)

    n = len(df)
    train_end = int(TRAIN_SPLIT * n)
    test_end = train_end + int(TEST_SPLIT * n)

    train_samples = df.iloc[:train_end].to_dict('records')
    test_samples = df.iloc[train_end:test_end].to_dict('records')
    val_samples = df.iloc[test_end:].to_dict('records')

    return train_samples, test_samples, val_samples

def get_dataloaders(benign_fraction=1.0, seed=None):
    """
    Create DataLoader objects for train, test, and validation sets.

    Args:
        benign_fraction (float): Fraction of benign images to include.
        seed (int, optional): A seed for reproducibility.

    Returns:
        tuple: (train_loader, test_loader, val_loader) as DataLoader objects.
    """
    image_root = "./ISIC_2020_Training_JPEG"
    csv_path = "./train-metadata.csv"
    train_samples, test_samples, val_samples = split_data(csv_path,
                                                    benign_fraction, seed=seed)

    train_loader = DataLoader(ISICDataset(image_root, train_samples),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=WORKERS)
    test_loader = DataLoader(ISICDataset(image_root, test_samples),
                             batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=WORKERS)
    val_loader = DataLoader(ISICDataset(image_root, val_samples),
                            batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=WORKERS)

    return train_loader, test_loader, val_loader