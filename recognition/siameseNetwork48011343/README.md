# Deep Metric Learning for Melanoma Detection with Siamese and Binary Classifier Networks

## Project Overview

Melanoma detection from dermoscopic images is a critical medical task due to the high mortality associated with late diagnosis. However, the ISIC 2020 dataset presents challenges such as class imbalance (few malignant cases) and visual similarity between benign and malignant lesions. To address this, the implemented approach combined deep metric learning and binary classification. This project implements a Siamese neural network with a ResNet 34 backbone that learns discriminative feature embeddings by comparing image triplets (anchor, positive, negative) and optimising a triplet loss, ensuring that embeddings of similar lesions are closer in the learned space. These embeddings are then passed into a binary classifier that predicts whether a lesion is benign or malignant. This joint setup allows the model to generalise effectively to new patient data, achieving an accuracy of greater than 0.8 on a given test set.

## Algorithm Description
This implemented algorithm follows a two-stage deep learning pipeline combining metric learning and supervised classification.

1. **Siamese Feature Extraction (Stage 1)**
   - A Siamese network based on ResNet 34 processes image triplets:
     - Anchor (a reference image)
     - Positive (same class as anchor)
     - Negative (different class)
   - The network passes each image through a shared feature extractor and a projection head, producing 128-dimensional embeddings. This projection head serves to transform backbone features         into a space optimised for triplet loss learning.
   - A triplet loss is applied, encouraging embeddings of the same class to be close together and embeddings of different classes to be far apart.
   - This stage learns a discriminative feature space, making lesions of the same type cluster together.

2. **Binary Classification (Stage 2)**
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
    - Loss function: Triplet margin loss (`TripletMarginLoss`) for training on triplets.
    - Optimiser: Adam optimiser applied to update Siamese network parameters.
    
  - **Training Strategy (triplet learning)**
    - Trained using triplet loss on *(anchor, positive, negative)* image sets.
    - Minimises distance between embeddings of similar lesions (benign-benign, malignant-malignant).
    - Maximises distance between embeddings of dissimilar lesions (benign-malignant).
    <img width="645" height="200" alt="triplet loss pic" src="https://github.com/user-attachments/assets/10f41aef-bf36-4939-931c-e439ff1bd065" />

      Triplet loss equation ($\alpha$ is the margin, ***a*** anchor, ***p*** positive, ***n*** negative):
   
    <img width="398" height="59" alt="triplet loss eq" src="https://github.com/user-attachments/assets/42f6dbd9-442e-48a8-a839-5a8af2739499" />

    A margin value of 1.0 was selected.

  - **Classifier Head**
    ![Classifier](https://github.com/user-attachments/assets/fca3990c-142a-477b-a635-ebc998a0b2f6)
    - Input: 128-dimensional embeddings from the Siamese network.
    - Architecture: Fully connected layers (128 &rarr; 512 &rarr; 256 &rarr; 64 &rarr; 2).
    - Activation: ReLU after each hidden layer.
    - Regularisation: Batch normalisation and dropout for generalisation.
    - Output: Two logits corresponding to benign and malignant cases.
    - Loss function: Cross-entropy loss is used to train the classifier, comparing predicted logits against ground-truth labels.
    - Optimiser: Adam optimiser is applied to update classifier parameters.
   
## Dependencies and Reproducibility
The main dependencies required are:
   - pandas: 2.3.2
   - numpy: 1.24.4
   - Pillow: 11.1.0
   - torch: 2.5.1
   - torchvision: 0.20.1
   - matplotlib: 3.10.5
   - scikit-learn: 1.7.1
   - tqdm: 4.67.1  

This project is designed to have reproducible results. All sources of randomness are seeded with `115` in `train.py`:  
```python
seed = 115
import random, numpy as np, torch

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
Doing this ensures reproducible results across:
   - Dataset splits (train/test/validaiton)
   - Triplet sampling in the Siamese dataset
   - Data augmentations (rotations, flip, colour jitter, affine transforms)
   - Model weight initialisation and training

## The Dataset
The pre-processed version of the dataset is used and can be found [here](https://www.kaggle.com/datasets/nischaydnk/isic-2020-jpg-224x224-resized). This one was used instead of the official version due it having a smaller image resolution of 256x256, which significantly reduced system storage.  

### Handling Class Imbalance
The dataset containing 33126 images was highly imbalanced. 32542 were of the benign class and the remaining 584 were malignant. To address this, targeted augmentation and triplet sampling strategies were applied.  

1. **Strong Data Augmentation for Malignant Images**

   Malignant samples were augmented with aggressive transformations to synthetically expand the minority class and improve model generalisation. The following augmentations were applied:
      - Random horizontal flips
      - Random vertical flips
      - Random rotations (up to 30 degrees)
      - Colour jitter (adjusting brightness, contrast, saturation, and hue)
      - Random affine (adds translation, scaling and shear distortions)

2. **Moderate Augmentation for Benign Images**

     Benign samples use a milder augmentation pipeline to avoid excessive distortion of the majority class:
      - Random horizontal flips
      - Random vertical flips
      - Random rotations (up to 15 degrees)

3. **Triplet Sampling**

   During training, each batch contained triplets as mentioned before. This ensures that benign and malignant samples appear in every batch, reinforcing discriminative feature learning between the two classes.

   See below for some sample data without augmentations followed by data with augmentations applied.
   <img width="650" height="450" alt="augs NONE" src="https://github.com/user-attachments/assets/5a2399f0-4119-48d2-ab8a-39b5f431ca5d" /> 
   <img width="650" height="450" alt="augs ON" src="https://github.com/user-attachments/assets/791770e5-eef1-4f90-8c12-7f74b0ad32d7" />

### Undersampling of the Majority Class Capabilities (not used in this project)
Despite undersampling not being used in the project, it should be noted that the parameter `benign_fraction` is present in `dataset.py`, which allows controlled undersampling of benign cases.
   - `benign_fraction = 1.0` &rarr; use all benign samples (default)
   - `benign_fraction < 1.0` &rarr; randomly selects a subset of benign images for a more balanced dataset

     For example, setting `benign_fraction=0.5` would use only half of the benign data while retaining all malignant cases. This drastically reduces runtime, but negatively affects generalisation due to a lower amount of diverse samples available.
     
It was decided to retain all benign cases (`benign_fraction = 1.0`), which effectively uses all 33126 images. This was decided because it would allow effective generalisation on unseen data (test set). However, this significantly greatly increases computational time as the model uses more data to train on as compared to if `benign_fraction < 1.0`.  

### Training, Validation, and Testing Splits
The dataset was randomly shuffled and split into three subsets.
   - Train Set: 70% of the data (ensures the model has enough examples to learn complex patterns from the data)
   - Test Set: 20% of the data (provides a substantial sample of unseen data, leading to a more reliable assessment of the model's ability to generalise)
   - Validation Set: 10% of the data (adequate amount of data for finding optimal classifiers producing highest validation accuracy after training)


    







