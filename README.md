# Galaxy Image classification using CNN
This repository contains the code developed for my Machine Learning course at University of Chicago. 

## Dataset
The dataset was made available via Kaggle (https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data). The dataset is produced by SDSS abd labeled via Galaxy Zoo. There are 61,578 labeled training images and 80,000 unlabeled test images.

## Architecture
The model was developed using ResNet34 architecture (no transfer learning) + some normalization and dropout layers. 

## Development
The code was developed on Google Colab (https://colab.research.google.com/)

## Training
The model was trainined on two Nvidia V100 GPUs on a Google Cloud Colab GCE VM using MirroredStratgy. Images were made available on the GCE VM to reduce the data loading time. 

## Accuracy
The model achieved  78.38% train, 79.03% validation and 78.32% test accuracy

Note: The Colab notebook contains implementation of Focal loss algorithm, based on the paper "Focal Loss for Dense Object Detection" by Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár.

## Focal Loss
Focal loss is designed to address the problem of class imbalance where the number of examples in one class significantly outnumbers the examples in other classes. In such cases, standard cross-entropy loss can cause the model to predominantly focus on the majority class, leading to poor model performance on the minority class. Focal loss modifies the cross-entropy loss to down-weight the loss assigned to well-classified examples, allowing the model to focus more on hard-to-classify instances which are generally few and more informative.
Parameters

**gamma (gamma factor)** : This parameter reduces the relative loss for well-classified examples (those with a high probability prediction), putting more focus on wrong classified examples. Higher values make the model focus even more on hard examples.
alpha (balancing factor): This parameter acts as a balancing factor for the classes. It can be used to give more importance to certain classes during training, typically the minority classes.
Components of the Function
Epsilon Handling:

**epsilon** is a small number added to predictions and subtracted from one minus predictions to ensure numerical stability of the log function in subsequent calculations.

### Prediction Clipping:
`y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)` ensures that predictions are within the range [epsilon, 1-epsilon] to avoid taking the logarithm of zero.

### Class Weights Calculation:

alpha_t adjusts the alpha values based on the true labels (y_true). If the true label is 1 (positive class), alpha_t is set to alpha. If the true label is 0 (negative class), alpha_t is set to 1 - alpha.
Modified Probability:

p_t calculates an adjusted probability for each class, based on the true labels. For the true class, it uses the predicted probability (y_pred), and for the false class, it uses 1 - y_pred.
Loss Calculation:

The loss for each example is calculated as - 

`alpha_t * tf.keras.backend.pow((tf.keras.backend.ones_like(y_true) - p_t), gamma) * tf.keras.backend.log(p_t)`. 

This formula applies a modulating factor (1 - p_t)^gamma to the cross-entropy loss to focus learning more on difficult examples.
tf.keras.backend.mean(fl) then takes the mean of these losses over all examples to compute the final loss value to optimize.


