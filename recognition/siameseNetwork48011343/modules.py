import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

class SiameseNetwork(nn.Module):
    """
    Siamese Network using a ResNet backbone + projection head.

    The backbone outputs a 512-dim feature, then the projection head
    refines it into a lower-dimensional, normalised embedding suitable
    for triplet loss training.
    """
    def __init__(self):
        super().__init__()
        # Feature extractor backbone
        self.feature_extractor = resnet34(weights=None)
        self.feature_extractor.fc = nn.Identity() # remove classification layer

        # Projection head (MLP)
        self.projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
        )

    def forward_once(self, x):
        feats = self.feature_extractor(x)
        proj = self.projection(feats)
        # Normalise embeddings to unit length
        proj = F.normalize(proj, p=2, dim=1)
        return proj

    def forward(self, x1, x2, x3):
        return (
            self.forward_once(x1),
            self.forward_once(x2),
            self.forward_once(x3)
        )

class BinaryClassifier(nn.Module):
    """
    Binary classifier for Siamese feature embeddings.

    Architecture:
        128 → 512 → 256 → 64 → 2
    Includes BatchNorm and Dropout to improve generalization.
    """

    def __init__(self, in_features=128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.layers(x)