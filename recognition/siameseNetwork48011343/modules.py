"""
modules.py

Defines the model architectures used for feature extraction and classification
of ISIC 2020 data.

1. SiameseNetwork:
    - Uses a ResNet 34 backbone to extract features.
    - A projection head converts features to lower-dimensional embeddings
      suitable for triplet loss training.
    - Embeddings are normalised to unit length.

2. BinaryClassifier:
    - Classifier operating on embeddings from the Siamese network.
    - Includes BatchNorm and Dropout to improve generalisation.
    - Outputs logits for 2-class classification (benign vs malignant).
"""

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

class SiameseNetwork(nn.Module):
    """
    Siamese Network using a ResNet backbone + projection head.

    The backbone outputs a 512-dim feature, then the projection head refines it
    into a lower-dimensional, normalised embedding suitable for triplet loss
    training.
    """

    def __init__(self):
        """
        Initialise Siamese Network.

        Components:
            - feature extractor: ResNet 34 without the final classification
              layer.
            - projection: MLP to map 512-dim features to 128-dim embeddings.
        """
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
        """
        Forward pass for a single input image.

        Args:
            x (Tensor): Input image tensor of shape [B, C, H, W]

        Returns:
            Tensor: Normalised embedding of shape [B, 128]
        """
        feats = self.feature_extractor(x)
        proj = self.projection(feats)

        # Normalise embeddings to unit length
        proj = F.normalize(proj, p=2, dim=1)
        return proj

    def forward(self, x1, x2, x3):
        """
        Forward pass for a triplet (anchor, positive, negative).

        Args:
            x1, x2, x3 (Tensor): Input tensors of shape [B, C, H, W]

        Returns:
            tuple: Normalised embeddings for each input (anchor, positive,
            negative)
        """
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
    Includes BatchNorm and Dropout to improve generalisation.
    """

    def __init__(self, in_features=128):
        """
        Initialise Binary Classifier.

        Args:
            in_features (int): Size of input embedding (default 128)
        """
        super().__init__()

        # Sequential MLP layers with BatchNorm, ReLU, and Dropout
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

            # Output logits for binary classification
            nn.Linear(64, 2)
        )

    def forward(self, x):
        """
        Forward pass of the classifier.

        Args:
            x (Tensor): Input embedding tensor of shape [B, in_features]

        Returns:
            Tensor: Logits tensor of shape [B, 2]
        """
        return self.layers(x)