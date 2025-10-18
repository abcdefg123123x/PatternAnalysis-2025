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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    # Load the test dataloader
    _, test_loader, _ = get_dataloaders()

    # Load trained models
    siamese_path = "checkpoints/best_siamese.pth"
    classifier_path = "checkpoints/best_classifier.pth"

    siamese = SiameseNetwork().to(device)
    classifier = BinaryClassifier().to(device)

    siamese.load_state_dict(torch.load(siamese_path, map_location=device))
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))

    siamese.eval()
    classifier.eval()

    # Feature extraction
    test_features, test_labels = extract_features_labels(siamese, test_loader,
                                                         device)

    # Evaluation
    print("Evaluating Binary Classifier on test embeddings:")

    cm = save_confusion_matrix(classifier, test_features, test_labels,
                        save_path="checkpoints/predict_confusion_matrix.png")

    # Compute ROC, AUC, Sensitivity, Specificity
    metrics = compute_roc_auc(classifier, test_features, test_labels,
                              save_path="checkpoints/predict_roc_curve.png")

    # Calculate test accuracy from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    test_acc = (tp + tn) / (tp + tn + fp + fn)
    print(f" Test Accuracy: {test_acc * 100:.2f}%")

    # Print summary
    print("\nEvaluation Summary:")
    print(f" AUC: {metrics['AUC']:.4f}")
    print(f" Sensitivity: {metrics['Sensitivity']:.4f}")
    print(f" Specificity: {metrics['Specificity']:.4f}")

    # t-SNE visualisation of embeddings
    print("\nGenerating t-SNE visualization for test embeddings...")
    plot_tsne(test_features, test_labels,
              save_path="checkpoints/predict_embeddings_tsne_predict.png",
              title="t-SNE of Test Embeddings (Predict Phase)")

    print("\nPrediction completed. Outputs saved in ./checkpoints/")

if __name__ == "__main__":
    main()