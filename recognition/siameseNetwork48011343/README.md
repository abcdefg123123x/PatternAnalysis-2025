# Deep Metric Learning for Melanoma Detection with Siamese and Binary Classifier Networks

## Project Overview

Melanoma detection from dermoscopic images is a critical medical task due to the high mortality associated with late diagnosis. However, the ISIC 2020 dataset presents challenges such as class imbalance (few malignant cases) and visual similarity between benign and malignant lesions. To address this, the implemented approach combined deep metric learning and binary classification. This project implements a Siamese neural network with a ResNet 34 backbone that learns discriminative feature embeddings by comparing image triplets (anchor, positive, negative) and optimising a triplet loss, ensuring that embeddings of similar lesions are closer in the learned space. These embeddings are then passed into a binary classifier that predicts whether a lesion is benign or malignant. This joint setup allows the model to generalise effectively to new patient data, achieving an accuracy of greater than 0.8 on a given test set.

## Algorithm Description
This implemented algorithm follows a two-stage deep learning pipeline combining metric learning and supervised classification.

1. **Siamese Feature Extraction (stage 1)**
   - A Siamese network based on ResNet 34 processes image triplets:
     - Anchor (a reference image)
     - Positive (same class as anchor)
     - Negative (different class)
   - The network passes each image through a shared feature extractor and a projection head, producing 128-dimensional embeddings. This projection head serves to transform backbone features         into a space optimised for triplet loss learning.
   - A triplet loss is applied, encouraging embeddings of the same class to be close together and embeddings of different classes to be far apart.
   - This stage learns a discriminative feature space, making lesions of the same type cluster together.

2. **Binary Classification (stage 2)**
   - The learned embeddings are fed into a binary classifier network (a multi-layer perception with BatchNorm and Dropout).
   - It outputs logits for benign and malignant classes.
   - This classifier is trained using standard cross-entropy loss, using embeddings from the trained Siamese model.
  
3. **Evaluation and Visualisation**
   - During prediction, the Siamese model generates embeddings for the validation subset.
   - The classifier then predicts class probabilities, and performance is measured using metrics such as accuracy, AUC, sensitivity, and specificity.
   - The learned feature is visualised using t-SNE, showing how benign and malignant images form distinct clusters.

### Network Architecture
  - **Embedding Network (Siamese backbone)**
    ![Siamese Network](https://github.com/user-attachments/assets/648f4776-fcf1-4edb-a690-1bb6915bf659)
    - Base model: ResNet 34 backbone pre-trained for general image feature extraction.
    - Projection head: Three fully connected layers (512 &rarr; 512 &rarr; 256 &rarr; 128).
    - Activation: ReLU applied after each layer.
    - Normalisation: L2 normalisation to constrain embeddings on the unit hypersphere.
    - Output: 128-dimensional embedding vector representing each input image.
  - **Training Strategy (triplet learning)**
    - Trained using triplet loss on *(anchor, positive, negative)* image sets.
    - Minimises distance between embeddings of similar lesions (benign-benign, malignant-malignant).
    - Maximises distance between embeddings of dissimilar lesions (benign-malignant).
  - **Classifier Head**
    ![Classifier](https://github.com/user-attachments/assets/fca3990c-142a-477b-a635-ebc998a0b2f6)
    - Input: 128-dimensional embeddings from the Siamese network.
    - Architecture: Fully connected layers (128 &rarr; 512 &rarr; 256 &rarr; 64 &rarr; 2).
    - Activation: ReLU after each hidden layer.
    - Regularisation: Batch normalisation and dropout for generalisation.
    - Output: Two logits corresponding to benign and malignant cases.



