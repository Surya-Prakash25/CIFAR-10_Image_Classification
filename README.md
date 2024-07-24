# CIFAR-10_Image_Classification


This repository contains a Jupyter notebook that implements a Convolutional Neural Network (CNN) using Keras to classify images from the CIFAR-10 dataset.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes. This project demonstrates how to build and train a CNN to classify these images.

## Dataset
The CIFAR-10 dataset includes the following classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

## Model Architecture
The model is built using the Keras Sequential API and includes the following layers:
- Convolutional layers
- MaxPooling layers
- Batch Normalization layers
- Fully connected (Dense) layers
- Dropout layer

## Training
The model is trained for a specified number of epochs with a batch size of 96 and uses the Adam optimizer.

## Results
The trained model is evaluated on the test set, and the predictions are compared with the true labels.

## Prerequisites
- Python 3.11.4
- Jupyter Notebook
- TensorFlow
- Keras
- Numpy
- Matplotlib

## Usage
1. Clone the repository:
    ```sh
    git clone https://github.com/Surya-Prakash25/CIFAR-10-Image-Classification.git
    cd CIFAR-10-Image-Classification
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Run the Jupyter notebook:
    ```sh
    jupyter notebook CIFAR10.ipynb
    ```

