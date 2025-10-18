import torch
from dataset import get_dataloaders
from modules import SiameseNetwork, BinaryClassifier
from utils import save_confusion_matrix, compute_roc_auc, plot_tsne

def extract_features_labels(model, loader, device):
    """Extracts feature embeddings and labels from the Siamese network."""
    feats_all, labels_all = [], []
    with torch.no_grad():
        for anchor, _, _, labels in loader:
            feats = model.forward_once(anchor.to(device))
            feats_all.append(feats)
            labels_all.append(labels.to(device))
    return feats_all, labels_all