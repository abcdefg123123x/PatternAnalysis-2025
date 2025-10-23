"""
utils.py

Utility functions for model evaluation, visualisation, and metric computation.

Includes:
    - Training/validation metric plotting
    - Confusion matrix generation and visualisation
    - t-SNE visualisation of learned feature embeddings
    - ROC/AUC computation with sensitivity and specificity metrics
"""

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
    Plot training and validation loss and accuracy over epochs.

    Args:
        train_losses (list[float]): Training loss per epoch.
        val_losses (list[float]): Validation loss per epoch.
        train_accs (list[float]): Training accuracy per epoch.
        val_accs (list[float]): Validation accuracy per epoch.
        save_path_prefix (str): Base path (without extension) to save figures.
        title (str): Model or experiment title to include in plot titles.

    Saves:
        - "<save_path_prefix>_loss.png": Loss curve.
        - "<save_path_prefix>_accuracy.png": Accuracy curve.
    """
    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)

    # Generate x-axis as epoch indicies
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
                          save_path="conf_matrix.png",
                          title="Test Set Confusion Matrix"):
    """
    Compute and save the confusion matrix for a trained classifier.

    Args:
        classifier (torch.nn.Module): Trained model used for prediction.
        features (list[torch.Tensor]): Batched input feature tensors.
        labels (list[torch.Tensor]): Batched ground truth label tensors.
        save_path (str): Path to save the confusion matrix plot.
        title (str): Title to display on the confusion matrix plot.
                     Defaults to "Test Set Confusion Matrix"

    Returns:
        np.ndarray: The computed confusion matrix.

    Saves:
        - A confusion matrix PNG to `save_path`.
    """
    classifier.eval()
    all_preds = []
    all_labels = []

    # Perform inference batch by batch
    with torch.no_grad():
        for feats, lbls in zip(features, labels):
            out = classifier(feats)
            preds = out.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(lbls.cpu())

    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Display and save confusion matrix as image
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

    # Print text version of matrix for quick inspection
    print(f"{title}:")
    print(cm)

    return cm

def plot_tsne(features_list, labels_list,
              save_path="checkpoints/tsne_plot.png", title="t-SNE Embeddings"):
    """
    Visualise high-dimensional feature embeddings using t-SNE.

    Args:
        features_list (list[torch.Tensor]): Batched feature tensors.
        labels_list (list[torch.Tensor]): Corresponding label tensors.
        save_path (str): Path to save the t-SNE visualisation.
        title (str): Title for the plot.

    Saves:
        - A t-SNE scatter plot of embeddings, colour-coded by class.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Merge all batches into single numpy arrays
    features_tensor = torch.cat(features_list).cpu().numpy()
    labels_tensor = torch.cat(labels_list).cpu().numpy()

    # Compute 2D t-SNE embeddings
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200,
                random_state=100)
    features_2d = tsne.fit_transform(features_tensor)

    # Plot the t-SNE projection
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
    Compute and plot the ROC curve, AUC, sensitivity, and specificity.

    Args:
        classifier (torch.nn.Module): Trained binary classifier.
        features (list[torch.Tensor]): Batched input feature tensors.
        labels (list[torch.Tensor]): Batched ground truth label tensors.
        save_path (str): Path to save the ROC plot.

    Returns:
        dict: {
            "AUC": float,
            "Sensitivity": float,
            "Specificity": float
        }

    Saves:
        - ROC curve PNG at `save_path`.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    classifier.eval()
    all_probs = []
    all_labels = []

    # Collect predicted probabilities for positive class
    with torch.no_grad():
        for feats, lbls in zip(features, labels):
            out = classifier(feats)
            probs = torch.softmax(out, dim=1)[:, 1]  # Pos class probability
            all_probs.append(probs.cpu())
            all_labels.append(lbls.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Compute ROC and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    # Compute sensitivity and specificity
    preds = (all_probs >= 0.5).astype(int)
    cm = confusion_matrix(all_labels, preds)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Save the plotted ROC curve
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

    # Print summary metrics
    print(f"\nROC and AUC Results:")
    print(f" - AUC: {roc_auc:.4f}")
    print(f" - Sensitivity (Recall for positive): {sensitivity:.4f}")
    print(f" - Specificity (True negative rate): {specificity:.4f}")

    return {
        "AUC": roc_auc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
    }