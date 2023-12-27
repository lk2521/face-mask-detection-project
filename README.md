# Face Mask Detection using Convolutional Neural Networks

This project focuses on training a Convolutional Neural Network (CNN) to classify whether a person in an image is wearing a face mask or not. The dataset used for this project is obtained from [Kaggle's Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset).

## Project Overview
The project follows these major steps:

### Data Retrieval and Preprocessing

* Downloading the dataset from Kaggle and extracting the contents.
* Exploring and preparing the dataset for model training.
### Image Processing

* Resizing images to a standard size (128x128) and converting them to numpy arrays for model input.
### Building the CNN Model

* Designing a CNN architecture using TensorFlow and Keras for mask detection.
### Model Training and Evaluation

* Splitting the dataset into training and testing sets.
* Scaling image data and training the CNN model.
* Evaluating model performance using validation and test sets.
### Predictive System

* Creating a system to predict mask-wearing status on new images.
## Requirements and Dependencies
* Python 3.12.1
* **Libraries**: TensorFlow, Keras, NumPy, Matplotlib, OpenCV
## Dataset Description
* The dataset comprises two classes:
* `images with masks` and `images without masks`.
* It consists of 3725 images with masks and 3828 images without masks.

## Model Architecture
The CNN architecture used for this classification task involves:

* **Input Layer**: Convolutional layers with kernel size (3x3) and ReLU activation.
* **Pooling Layers**: Max-pooling layers for downsampling.
* **Flattening Layer**: Flattening the output for fully connected layers.
* **Dense Layers**: Fully connected layers with dropout for regularization.
* **Output Layer**: Sigmoid activation to predict mask presence (1) or absence (0).
