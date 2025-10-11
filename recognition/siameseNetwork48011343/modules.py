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