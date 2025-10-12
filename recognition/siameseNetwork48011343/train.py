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

# Training functions
def train_siamese(model, train_loader, val_loader, loss_fn, optimiser,
                  epochs, device):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Siamese Epoch {epoch+1}/{epochs}")
        for anchor, pos, neg, _ in loop:
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
            optimiser.zero_grad()
            a_feat, p_feat, n_feat = model(anchor, pos, neg)
            loss = loss_fn(a_feat, p_feat, n_feat)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
            loop.set_postfix(loss=total_loss / (loop.n + 1))

        train_losses.append(total_loss / len(train_loader))
        train_acc = compute_siamese_accuracy(model, train_loader, device)
        val_loss = compute_siamese_val_loss(model, val_loader, loss_fn, device)
        val_acc = compute_siamese_accuracy(model, val_loader, device)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Siamese Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.4f} - "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return train_losses, val_losses, train_accs, val_accs

def train_classifier(classifier, features_train, labels_train,
                     features_val, labels_val, loss_fn, optimiser, epochs):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        classifier.train()
        total_loss, correct_train, total_train = 0, 0, 0
        loop = tqdm(zip(features_train, labels_train),
                    total=len(features_train),
                    desc=f"Classifier Epoch {epoch+1}/{epochs}")
        for feats, labels in loop:
            optimiser.zero_grad()
            out = classifier(feats) # features from Siamese network
            loss = loss_fn(out, labels)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
            preds = out.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            loop.set_postfix(loss=total_loss / (loop.n + 1),
                             acc=correct_train / total_train)

        val_loss, val_acc = evaluate_classifier(classifier, features_val,
                                                labels_val, loss_fn)
        train_losses.append(total_loss / len(features_train))
        val_losses.append(val_loss)
        train_accs.append(correct_train / total_train)
        val_accs.append(val_acc)

        print(f"Classifier Epoch {epoch + 1}/{epochs} - "
              f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.4f} - "
              f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accs[-1]:.4f}")

    return train_losses, val_losses, train_accs, val_accs