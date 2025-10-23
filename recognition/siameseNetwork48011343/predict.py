"""
predict.py

Runs inference and evaluation using trained Siamese and binary classifier models
on a subset of the ISIC 2020 dataset.

This script:
    - Loads trained model weights from the `checkpoints/` directory.
    - Extracts embeddings from a Siamese feature extractor.
    - Evaluates classification performance using confusion matrix, ROC/AUC,
      sensitivity, and specificity.
    - Generates a t-SNE visualisation of learned embeddings.

Notes:
    The script uses the **validation dataloader** as a *sample subset* of the
    full dataset for prediction and evaluation. This means metrics are computed
    on a representative validation sample rather than the full dataset or a
    dedicated held-out test set.
"""

import torch
from dataset import get_dataloaders
from modules import SiameseNetwork, BinaryClassifier
from utils import save_confusion_matrix, compute_roc_auc, plot_tsne

def extract_features_labels(model, loader, device):
    """
    Extract feature embeddings and labels from a dataloader using a Siamese network.

    Args:
        model (nn.Module): Trained Siamese model (feature extractor).
        loader (DataLoader): DataLoader providing image triplets and labels.
        device (torch.device): Target device for computation.

    Returns:
        tuple[list[Tensor], list[Tensor]]:
            - feats_all: list of feature tensors (embeddings per batch)
            - labels_all: list of label tensors
    """
    feats_all, labels_all = [], []

    # Disable gradient computation during inference
    with torch.no_grad():
        for anchor, _, _, labels in loader:
            # Forward pass only through the anchor branch of the Siamese net
            feats = model.forward_once(anchor.to(device))
            feats_all.append(feats)
            labels_all.append(labels.to(device))

    return feats_all, labels_all

def main():
    """
    Main entry point for prediction and evaluation.

    Loads pre-trained Siamese and Binary Classifier models, performs inference
    on the validation (sample) dataset, computes key metrics, and generates plots.

    Outputs:
        - Confusion matrix (PNG)
        - ROC curve (PNG)
        - t-SNE embedding plot (PNG)
        - Printed AUC, sensitivity, specificity, and test accuracy
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    # Load the sample subset for evaluation
    # Note: This uses the validation loader, not a full test set.
    _, _, sample_loader = get_dataloaders()

    # Load the pre-trained model weights
    siamese_path = "checkpoints/best_siamese.pth"
    classifier_path = "checkpoints/best_classifier.pth"

    siamese = SiameseNetwork().to(device)
    classifier = BinaryClassifier().to(device)

    siamese.load_state_dict(torch.load(siamese_path, map_location=device))
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))

    siamese.eval()
    classifier.eval()

    # Feature extraction from the Siamese model
    predict_features, predict_labels = extract_features_labels(siamese,
                                                        sample_loader, device)

    # Evaluate classifier performance from here onwards
    print("Evaluating Binary Classifier on test embeddings:")

    # Confusion matrix
    cm = save_confusion_matrix(classifier, predict_features, predict_labels,
                        save_path="checkpoints/PREDICT_confusion_matrix.png",
                               title="Sample Set Confusion Matrix")

    # ROC, AUC, Sensitivity, Specificity
    metrics = compute_roc_auc(classifier, predict_features, predict_labels,
                              save_path="checkpoints/PREDICT_roc_curve.png")

    # Calculate test (validation) accuracy from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    test_acc = (tp + tn) / (tp + tn + fp + fn)
    print(f" Test Accuracy: {test_acc * 100:.2f}%")

    # Print summary metrics
    print("\nEvaluation Summary:")
    print(f" AUC: {metrics['AUC']:.4f}")
    print(f" Sensitivity: {metrics['Sensitivity']:.4f}")
    print(f" Specificity: {metrics['Specificity']:.4f}")

    # Visualise embeddings using t-SNE
    print("\nGenerating t-SNE visualisation for test embeddings...")
    plot_tsne(predict_features, predict_labels,
              save_path="checkpoints/PREDICT_embeddings_tsne_predict.png",
              title="t-SNE of Test Embeddings (Predict Phase)")

    print("\nPrediction completed. Outputs saved in ./checkpoints/")

if __name__ == "__main__":
    main()