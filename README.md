# Waste Classification with VGG16

This project aims to classify waste images into two categories: Organic and Recycle waste using a pre-trained VGG16 model fine-tuned for the task.

## Table of Contents
- [Introduction](#introduction)
- [Problem Statement](#ProblemStatement)
- [Model Overview](#ModelOverview)
- [Dataset]
- [Model Architecture]
- [Results]
- [License]

# Introduction

Waste Classification is a computer vision project that leverages a VGG16-based deep learning model to classify waste images. The project focuses on distinguishing between Organic and Recycle waste, making it a valuable tool for waste management and environmental initiatives.

# Problem Statement

The waste classification problem is a critical aspect of waste management. Properly identifying and classifying waste materials can help in recycling and disposal processes, reducing environmental impact.

# Model Overview

This project utilizes a fine-tuned VGG16-based CNN model to classify waste materials. The model architecture includes custom dense layers with batch normalization, ReLU activation, and dropout layers to improve performance. Here's a brief overview of the model architecture:

- VGG16-based feature extraction layer (pre-trained on ImageNet)
- Dropout layer with a 20% dropout rate
- Flattening layer
- Batch normalization layers
- Dense layers with 1024, 512, 256, 512, and 512 filters, respectively
- Activation layers (ReLU)
The final output layer is a single neuron with a sigmoid activation function for binary classification.

# Dataset
Problem: Waste management challenges, including landfill overflow and pollution.
Approach: Analyzed waste components, segregated into Organic and Recyclable using IoT and ML.
Implementation: Dataset split - 85% training (22,564 images) and 15% testing (2,513 images).
You can access the complete dataset on <a href="https://www.kaggle.com/datasets/techsash/waste-classification-data" target="blank">Kaggle</a>.





