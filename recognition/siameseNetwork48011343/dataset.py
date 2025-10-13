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
            return self.transform(img)

        return (load_img(anchor), load_img(positive), load_img(negative),
                anchor['target'])

def split_data(csv_path, benign_fraction=0.2, seed=None):
    """
    Split the dataset while using a fraction of benign images.

    Arguments:
        csv_path: path to CSV metadata
        benign_fraction: float (0,1] - fraction of benign images to keep
        seed: int - random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
        pd.np.random.seed(seed)

    df = pd.read_csv(csv_path)
    malignant = df[df['target'] == 1].copy()  # keep all malignant images
    benign = df[
        df['target'] == 0].copy()  # keep all benign initially

    # Keep only a fraction of benign images
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


def get_dataloaders(benign_fraction=0.2, seed=None):
    """
    Returns train, test, val dataloaders.
    benign_fraction: fraction of benign images to use in training
    seed: random seed for reproducibility
    """
    image_root = "./ISIC_2020_Training_JPEG"
    csv_path = "./train-metadata.csv"
    train_samples, test_samples, val_samples = split_data(csv_path,
                                                    benign_fraction, seed=seed)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    train_loader = DataLoader(ISICDataset(image_root, train_samples, transform),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=WORKERS)
    test_loader = DataLoader(ISICDataset(image_root, test_samples, transform),
                             batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=WORKERS)
    val_loader = DataLoader(ISICDataset(image_root, val_samples, transform),
                            batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=WORKERS)

    return train_loader, test_loader, val_loader