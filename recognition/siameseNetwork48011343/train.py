import torch
from torch.nn import TripletMarginLoss, CrossEntropyLoss
from torch.optim import Adam
from modules import SiameseNetwork, BinaryClassifier
from dataset import get_dataloaders
from tqdm import tqdm

# Hyperparameters
EPOCHS_SIAMESE = 40
EPOCHS_CLASSIFIER = 25
LR_SIAMESE = 1e-4
LR_CLASSIFIER = 5e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helper functions
def compute_siamese_accuracy(model, loader, device):
    """Fraction of triplets where anchor is closer to positive than negative"""
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for anchor, pos, neg, _ in loader:
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
            a_feat, p_feat, n_feat = model(anchor, pos, neg)
            pos_dist = (a_feat - p_feat).pow(2).sum(dim=1)
            neg_dist = (a_feat - n_feat).pow(2).sum(dim=1)
            correct += (pos_dist < neg_dist).sum().item()
            total += anchor.size(0)
        return correct / total

def compute_siamese_val_loss(model, loader, loss_fn, device):
    """Calculates validation triplet loss."""
    model.eval()
    val_loss_total = 0
    with torch.no_grad():
        for anchor, pos, neg, _ in loader:
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
            a_feat, p_feat, n_feat = model(anchor, pos, neg)
            val_loss_total += loss_fn(a_feat, p_feat, n_feat).item()
    return val_loss_total / len(loader)

def extract_features_labels(model, loader, device):
    """Extract embeddings and labels from the Siamese network."""
    features_list, labels_list = [], []
    model.eval()
    with torch.no_grad():
        for anchor, _, _, labels in loader:
            feats = model.forward_once(anchor.to(device))
            features_list.append(feats)
            labels_list.append(labels.to(device))
    return features_list, labels_list


def evaluate_classifier(classifier, features, labels, loss_fn):
    """Compute classifier loss and accuracy."""
    classifier.eval()
    total_loss = 0
    correct, total = 0, 0
    with torch.no_grad():
        for feats, lbls in zip(features, labels):
            out = classifier(feats)
            total_loss += loss_fn(out, lbls).item()
            preds = out.argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)
    return total_loss / len(features), correct / total