import os
import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc)

def plot_metrics(train_losses, val_losses, train_accs, val_accs,
                 save_path_prefix="checkpoints/metrics", title="Model"):
    """
    Plot training/validation loss and accuracy curves and save as PNG.
    """
    os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)

    epochs = np.arange(1, len(train_losses) + 1)

    # Plot Loss
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path_prefix}_loss.png")
    plt.close()

    # Plot Accuracy
    plt.figure()
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, val_accs, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title} Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path_prefix}_accuracy.png")
    plt.close()

def save_confusion_matrix(classifier, features, labels,
                          save_path="conf_matrix.png"):
    """Compute, save, and print confusion matrix for classifier."""
    classifier.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for feats, lbls in zip(features, labels):
            out = classifier(feats)
            preds = out.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(lbls.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Save as an image
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Test Set Confusion Matrix")
    plt.savefig(save_path)
    plt.close()  # Close to free memory

    # Print a text version
    print("Confusion Matrix (test set):")
    print(cm)
    return cm


def plot_tsne(features_list, labels_list,
              save_path="checkpoints/tsne_plot.png", title="t-SNE Embeddings"):
    """
    Plot t-SNE of embeddings and save as PNG.

    features_list: list of torch tensors (embeddings)
    labels_list: list of torch tensors (labels)
    save_path: file path to save the figure
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Combine batches
    features_tensor = torch.cat(features_list).cpu().numpy()
    labels_tensor = torch.cat(labels_list).cpu().numpy()

    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200,
                random_state=100)
    features_2d = tsne.fit_transform(features_tensor)

    # Plot
    plt.figure(figsize=(8, 8))
    for label in np.unique(labels_tensor):
        idx = labels_tensor == label
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1],
                    label=f"Class {label}", alpha=0.6)
    plt.legend()
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def compute_roc_auc(classifier, features, labels,
                    save_path="checkpoints/roc_curve.png"):
    """
    Compute ROC curve, AUC, sensitivity, and specificity.
    Saves ROC plot and prints metrics.
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    classifier.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for feats, lbls in zip(features, labels):
            out = classifier(feats)
            probs = torch.softmax(out, dim=1)[:, 1]  # probability for pos class
            all_probs.append(probs.cpu())
            all_labels.append(lbls.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # ROC and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    # Compute confusion matrix for sensitivity/specificity
    preds = (all_probs >= 0.5).astype(int)
    cm = confusion_matrix(all_labels, preds)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Save ROC plot
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

    print(f"\nROC and AUC Results:")
    print(f" - AUC: {roc_auc:.4f}")
    print(f" - Sensitivity (Recall for positive): {sensitivity:.4f}")
    print(f" - Specificity (True negative rate): {specificity:.4f}")

    return {
        "AUC": roc_auc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
    }