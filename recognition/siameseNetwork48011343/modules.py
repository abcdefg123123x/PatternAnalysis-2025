import torch.nn as nn
from torchvision.models import resnet50

class SiameseNetwork(nn.Module):
    """
    Siamese Network using a ResNet50 backbone for feature extraction.

    Replaces the final classificaiton layer with identity to output 2048-dim
    feature embeddings. Used for triplet loss training.
    """
    def __init__(self):
        super().__init__()
        self.feature_extractor = resnet50(weights=None)
        self.feature_extractor.fc = nn.Identity() # output embedding

    def forward_once(self, x):
        return self.feature_extractor(x)

    def forward(self, x1, x2, x3):
        return (self.forward_once(x1), self.forward_once(x2),
                self.forward_once(x3))

class BinaryClassifier(nn.Module):
    """
    Binary classifier for features from Siamese network.

    Takes 2048-dim feature embeddings and classifies them as benign (0) or
    malignant (1). Includes dropout for regularisation.

    Architecture: 2048 -> 1024 -> 256 -> 2
    """
    def __init__(self, in_features=2048):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.layers(x)