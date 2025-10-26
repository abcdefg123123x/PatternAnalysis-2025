# Deep Metric Learning for Melanoma Detection with Siamese and Binary Classifier Networks

## Project Overview

Melanoma detection from dermoscopic images is a critical medical task due to the high mortality associated with late diagnosis. However, the ISIC 2020 dataset presents challenges such as class imbalance (few malignant cases) and visual similarity between benign and malignant lesions. To address this, the implemented approach combined deep metric learning and binary classification. This project implements a Siamese neural network with a ResNet 34 backbone that learns discriminative feature embeddings by comparing image triplets (anchor, positive, negative) and optimising a triplet loss, ensuring that embeddings of similar lesions are closer in the learned space. These embeddings are then passed into a binary classifier that predicts whether a lesion is benign or malignant. This joint setup allows the model to generalise effectively to new patient data, achieving an accuracy of greater than 0.8 (80%) on a given test set.

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
   - Higher logits indicate greater model confidence for the corresponding class. These logits are later converted into probabilities using the softmax function during evaluation.
   - This classifier is trained using standard cross-entropy loss, using embeddings from the trained Siamese model.
  
3. **Evaluation and Visualisation**
   - During prediction, the Siamese model generates embeddings for the validation subset.
   - The classifier then predicts class probabilities, and performance is measured using metrics such as accuracy, AUC, sensitivity, and specificity.
   - The learned feature is visualised using t-SNE, showing how benign and malignant images form distinct clusters.

### Network Architecture
  - **Embedding Network (Siamese backbone)**
    ![Siamese Network](https://github.com/user-attachments/assets/766b2719-036e-4cbd-9cbd-ec9bb376d804)

    - Base model: ResNet 34 backbone pre-trained for general image feature extraction.
    - Projection head: Three fully connected layers (512 &rarr; 512 &rarr; 256 &rarr; 128).
    - Activation: ReLU applied after each layer.
    - Normalisation: L2 normalisation to constrain embeddings on the unit hypersphere.
    - Output: 128-dimensional embedding vector representing each input image.
    - Loss function: Triplet margin loss (`TripletMarginLoss`) for training on triplets.
    - Optimiser: Adam optimiser applied to update Siamese network parameters (learning rate of $1 \times 10^{-4}$).
    - Epochs: 30
    
  - **Training Strategy (triplet learning)**
    - Trained using triplet loss on *(anchor, positive, negative)* image sets.
    - Minimises distance between embeddings of similar lesions (benign-benign, malignant-malignant).
    - Maximises distance between embeddings of dissimilar lesions (benign-malignant).
    <img width="645" height="200" alt="triplet loss pic" src="https://github.com/user-attachments/assets/10f41aef-bf36-4939-931c-e439ff1bd065" />

      Triplet loss equation ($\alpha$ is the margin, ***a*** anchor, ***p*** positive, ***n*** negative, *d()* is the Euclidean distance):
   
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
    - Optimiser: Adam optimiser is applied to update classifier parameters (learning rate of $5 \times 10^{-4}$).
    - Epochs: 25
   
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
Doing this ensures the seed is applied globally across all files, including `dataset.py`, as long as `train.py` sets the seed before importing or using other modules.

This ensures reproducible results across:
   - Dataset splits (train/test/validation)
   - Triplet sampling in the Siamese dataset
   - Data augmentations (rotations, flip, colour jitter, affine transforms)
   - Model weight initialisation and training

If a different seed is to be used, simple replace the value of `seed` with another integer. Alternatively, to use a completely random seed each run, simply generate one with: 
```python
seed = random.randint(0, 2**32 - 1)
```

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

### Capabilities for Undersampling the Majority Class (not used in this project)
Despite undersampling not being used in the project, it should be noted that the parameter `benign_fraction` is present in `dataset.py`, which allows controlled undersampling of benign cases. This was initially implemented and was considered for use because undersampling prevents the model from being biased towards the majority class and improves its ability to predict the minority class.
   - `benign_fraction=1.0` &rarr; use all benign samples (default)
   - `benign_fraction<1.0` &rarr; randomly selects a subset of benign images for a more balanced dataset

     For example, setting `benign_fraction=0.5` would use only half of the benign data while retaining all malignant cases. This drastically reduces runtime, but negatively affects generalisation due to a lower amount of diverse samples available.
     
It was decided to retain all benign cases (`benign_fraction=1.0`), which effectively uses all 33126 images. This was decided because it would allow effective generalisation on unseen data (test set). However, this greatly increases computational time as the model uses more data to train on as compared to if `benign_fraction<1.0`.  

### Training, Validation, and Testing Splits
The dataset was randomly shuffled and split into three subsets.
   - Train Set: 70% of the data (ensures the model has enough examples to learn complex patterns from the data)
   - Test Set: 20% of the data (provides a substantial sample of unseen data, leading to a more reliable assessment of the model's ability to generalise)
   - Validation Set: 10% of the data (adequate amount of data for finding optimal classifiers producing highest validation accuracy after training)

## Usage
Firstly, ensure that all 33126 images are stored inside a folder named: `ISIC_2020_Training_JPEG/`  
The CSV file containing the metadata must also be named: `train-metadata.csv`  

Ensure the files are organised as follows:

```
├── checkpoints/
├── ISIC_2020_Training_JPEG/
│ ├── ISIC_0015719.jpg
│ ├── ISIC_0052212.jpg
│ ├── ISIC_0068279.jpg
│ └── ...
├── train-metadata.csv
├── dataset.py
├── modules.py
├── train.py
├── predict.py
├── utils.py
├── README.md
```
Ensure all dependencies are installed as listed in the [Dependencies and Reproducibility](#dependencies-and-reproducibility) section.  

To perform training and testing of the models, simply run the following:  

```
python train.py
```  

This will print out a log of the training, validation, and testing information like the following (most of it is omitted due to its length):  
```
Using the following seed for reproducibility: 115
Siamese Epoch 1/30: 100%|██████████| 725/725 [04:52<00:00,  2.48it/s, loss=0.123]
Saved best Siamese model with validation accuracy: 0.6450
Siamese Epoch 1/30 - Train Loss: 0.1230, Train Acc: 0.6509 - Val Loss: 0.8900, Val Acc: 0.6450
Siamese Epoch 2/30: 100%|██████████| 725/725 [04:57<00:00,  2.44it/s, loss=0.00863]
Saved best Siamese model with validation accuracy: 0.7160
Siamese Epoch 2/30 - Train Loss: 0.0086, Train Acc: 0.7050 - Val Loss: 0.8313, Val Acc: 0.7160
⋮
Siamese Epoch 30/30: 100%|██████████| 725/725 [04:40<00:00,  2.58it/s, loss=3.01e-6]
Siamese Epoch 30/30 - Train Loss: 0.0000, Train Acc: 0.6917 - Val Loss: 0.8589, Val Acc: 0.6822
Classifier Epoch 1/25: 100%|██████████| 725/725 [00:01<00:00, 591.86it/s, acc=0.991, loss=0.0349]
Saved best Classifier model with validation accuracy: 0.9934
Classifier Epoch 1/25 - Train Loss: 0.0345, Train Acc: 0.9906 - Val Loss: 0.0218, Val Acc: 0.9934
Classifier Epoch 2/25: 100%|██████████| 725/725 [00:01<00:00, 596.06it/s, acc=0.993, loss=0.0214]
Saved best Classifier model with validation accuracy: 0.9949
Classifier Epoch 2/25 - Train Loss: 0.0212, Train Acc: 0.9934 - Val Loss: 0.0201, Val Acc: 0.9949
⋮
Classifier Epoch 25/25 - Train Loss: 0.0122, Train Acc: 0.9962 - Val Loss: 0.0188, Val Acc: 0.9958
Test Accuracy: 99.47%
Test Set Confusion Matrix:
Labels: ['Benign', 'Malignant']
[[6494   19]
 [  16   96]]

ROC and AUC Results:
 - AUC: 0.9973
 - Sensitivity (Recall for positive): 0.8571
 - Specificity (True negative rate): 0.9971

Training completed. All model weights, evaluation metrics, and embedding visualisations have been saved in the ./checkpoints/ directory.
```
The printed text version of the confusion matrix corresponds to the following in the [Test Set Confusion Matrix](#test-set-confusion-matrix) of the results section.

Note that the loss and accuracy plots (for train and validation sets), t-SNE embeddings (for all three sets), test set confusion matrices, and test set ROC curve are saved as images in the directory `checkpoints/`.  

After training (`train.py`), the `checkpoints/` directory will contain `best_siamese.pth` and `best_classifier.pth`. These files store each model's weights corresponding to the highest validation accuracy achieved during training. These two files are also required for inferencing with `predict.py`. Without them, the script cannot load the trained models to make predictions.  

To perform inference on trained models, simply run the following:  
```
python predict.py
```
Note that `predict.py` is intended for demonstration purposes only on the trained models. It uses a sample subset of the dataset (10% of all images) for evaluation, drawn randomly from the validation dataloader. The sampled images may or may not have been seen during training, so results on this subset do not necessarily reflect performance on a dedicated test set (the test set was already done in `train.py` as mentioned before). Each run may produce a different set of images due to random sampling.  

Running inference will produce (the following is an example, and is different each run):  
```
Running inference on device: cuda
Evaluating Binary Classifier on sample embeddings:
Sample Set Confusion Matrix:
Labels: ['Benign', 'Malignant']
[[3242    5]
 [  10   56]]

ROC and AUC Results:
 - AUC: 0.9992
 - Sensitivity (Recall for positive): 0.8485
 - Specificity (True negative rate): 0.9985
 Accuracy on sample data: 99.55%

Evaluation Summary:
 AUC: 0.9992
 Sensitivity: 0.8485
 Specificity: 0.9985

Generating t-SNE visualisation for sample embeddings...

Prediction completed. Outputs saved in ./checkpoints/
```
The confusion matrix, ROC curve, and t-SNE embedding of the sample set are saved as images in `checkpoints/`. Outputs of these can be viewed in the [Results of Sample Data](#results-of-sample-data-predictpy-outputs) section. Check this section for the confusion matrix labels.

## Results
### Siamese Network Results
#### Loss Plot
<img width="540" height="380" alt="siamese_metrics_loss" src="https://github.com/user-attachments/assets/4856bbdd-3198-4f63-ba11-0b63b643e796" />  

The Siamese network training loss drops sharply from 0.123 in epoch 1 to near zero by epoch 6, indicating that the network quickly satisfies the triplet constraints on the training set. In constrast, the validation loss remains relatively high at 0.89 in epoch 1, decreasing to around 0.598 by epoch 8, then flucuating between 0.65 and 0.88. This shows that the network overfits the training triplets and struggles to generalise to unseen data.

#### Accuracy Graph
<img width="540" height="380" alt="siamese_metrics_accuracy" src="https://github.com/user-attachments/assets/315544dc-671e-480b-b23a-3716a2c5adea" />  

Both training and validation accuracies share very similar trends. The accuracies improve significantly from around 65% at epoch 1 to over 87% at other epochs. However, the accuracies remain lower after epoch 15, fluctuating between 55% and 83%. This indicates potential instability in the later training stages. This means the model maintains good generalisation, but may be experiencing optimisation challenges or convergence issues in the final epochs.

#### Training Data t-SNE Scatterplot
<img width="450" height="450" alt="train_embeddings_tsne" src="https://github.com/user-attachments/assets/b8aa7cd2-d974-4f83-9f1d-97bc8a8efff5" />  

'Class 0' corresponds to the benign class, and 'Class 1' corresponds to the malignant. This is also the same for all other t-SNE scatterplots.

The t-SNE plot shows the Siamese network has learned some class separation, but with noticeable overlap between benign and malignant clusters. This indicates the model captures meaningful features but struggles to fully distinguish between lesion types, explaining occasional misclassifications despite overall good performance.

#### Validation and Test Data t-SNE Scatterplot
<img width="450" height="450" alt="val_embeddings_tsne" src="https://github.com/user-attachments/assets/a7a466f7-7eff-4b56-a987-bcbdc3ca51c1" /> <img width="450" height="450" alt="test_embeddings_tsne" src="https://github.com/user-attachments/assets/f27c9d82-8c37-4a78-aae6-843de61edf0b" />  

The validation and test embeddings show clearer class separation than the training set, with more distinct benign and malignant clusters. This indicates the Siamese network generalises well to unseen data and learns meaningful features that transfer effectively beyond the training samples.

### Binary Classifier Results
#### Loss Plot
<img width="540" height="380" alt="classifier_metrics_loss" src="https://github.com/user-attachments/assets/675fc8fb-fded-406a-b894-682e1b178c98" />  

The classifier's training loss decreases rapidly during the first few epochs, from around 0.035 in epoch 1 to roughly 0.015 by epoch 10. This continues to decline gradually, reaching about 0.012 by epoch 25. The validation loss closely follows this trend, remaining consistently low (around 0.016-0.020) with only minor fluctuations across epochs. This indicates that the classifier quickly converged and maintained stable generalisation without significant overfitting. 

#### Accuracy Graph
<img width="540" height="380" alt="classifier_metrics_accuracy" src="https://github.com/user-attachments/assets/12590667-bf5b-4edd-80f9-f659eb9a5fb3" />  

The accuracies rise sharply during the initial epochs, improving from around 99% in epoch 1 to approximately 99.5% by epoch 5. After this rapid increase, both training and validation accuracies stabilise, consistently remaining between 99.4% and 99.7% for the remainder of training. This indicates that the model quickly converged and maintained strong, stable performance without signs of overfitting.

#### Test Set ROC Curve
<img width="540" height="380" alt="test_roc_curve" src="https://github.com/user-attachments/assets/b81bb970-b2a1-4590-86cf-590a08f25b1d" />  

The AUC is actually 0.9973 (says 1.0 in the image due to plot rounding and display precision). This ROC curve and AUC value shows excellent separability between benign and malignant classes in the test set. The curve closely approaches the top-left corner, reflecting minimal overlap between the predicted probabilities of the two categories.

#### Test Set Confusion Matrix
<img width="540" height="380" alt="test_conf_matrix" src="https://github.com/user-attachments/assets/d5f547ab-610a-4dac-9f33-76ec87d6c1de" />

#### Test Performance
   - Test Accuracy: `99.47%`
   - Sensitivity (Recall for positive): `0.8571`
   - Specificity (True negative rate): `0.9971`

The model achieved 99.47% test accuracy, substantially exceeding the project's target performance of 80% accuracy. The model achieves a high sensitivity of 0.8571, which reflects good detection capability. However, further improvement may be desirable, as medical screening tasks typically emphasise maximising sensitivity to minimise the risk of missed diagnoses.

## Results of Sample Data (`predict.py` outputs)
This section shows the outputs from `predict.py` when the sample data is fed into the entire network for evaluation or prediction. The network uses the saved optimal classifier weights obtained during training (from `train.py`), which effectively shows example usage of the entire trained model. Due to this, no analysis will be made on the outputs and they are presented for demonstration purposes only. Additionally, analysis is omitted because this is not a dedicated test set on completely unseen data as mentioned before.

### t-SNE Scatterplot and ROC Curve
<img width="400" height="400" alt="PREDICT_embeddings_tsne_predict" src="https://github.com/user-attachments/assets/0783a5cd-5dce-497a-8c19-711f3d815ae7" /> <img width="500" height="400" alt="PREDICT_roc_curve" src="https://github.com/user-attachments/assets/bc09941e-4201-4dcf-885e-6fa3c49f64a4" />

The actual AUC was 0.9992.

### Confusion Matrix
<img width="540" height="380" alt="PREDICT_confusion_matrix" src="https://github.com/user-attachments/assets/e9aae9f5-e85e-461d-af6f-f19da635a479" />

### Sample Data Performance
   - Accuracy: `99.55%`
   - Sensitivity: `0.8485`
   - Specificity: `0.9985`

## Future Recommendations
Based on the test performance results from the trained models (results of `train.py`), some recommendations for improvement can be considered. These recommendations focus on increasing the sensitivity value because it is important for the entire model to accurately identify true malignant cases in a real medical setting.
1. **Weighted Cross-Entropy**

   Weighted cross-entropy multiplies the loss for each class by a class-specific weight. For a highly imbalanced dataset like ISIC 2020, assigning a higher weight to the malignant class forces the binary classifier to focus more on correctly identifying malignant samples. This typically increases sensitivity but may slightly reduce specificity. Careful tuning of the class weights is important to achieve a balanced trade-off between the two.

2. **Focal Loss**

   Focal loss is an alternative to cross-entropy that dynamically down-weights easy examples and focuses the binary classifier's learning on hard or misclassified samples. This is particularly useful for this imbalanced dataset, where benign images dominate. By reducing the contribution of easily classified benign samples to the loss, the model becomes more sensitive to malignant cases. The focusing parameter ($\gamma$) controls how much harder samples are emphasised and should be tuned carefully for optimal performance.

3. **Data Augmentation with Synthetic Oversampling**

   Techniques such as SMOTE (Synthetic Minority Oversampling Technique) or GAN-based data generation can be applied to increase the number and diversity of malignant images. This helps the classifier learn more about robust malignant features and reduces overfitting to the malignant samples. However, synthetic data must be carefully validated to ensure it represents realistic lesion patterns and does not introduce artificial bias.

## References


