# Human Action Recognition using CNN-LSTM

## Introduction
This project presents an approach to Human Action Recognition (HAR) using a deep learning model that combines Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks. The model is designed to capture both the spatial features from individual video frames and the temporal dynamics across sequences of frames.

## Project Structure
- `temporal_transform.py`: Functions for temporal preprocessing of video data.
- `spatial_transform.py`: Functions for spatial preprocessing of video frames.
- `target_transform.py`: Functions for transforming labels and annotations.
- `data_utils.py`: Utilities for data loading and preprocessing.
- `cnnlstm.py`: The CNN-LSTM model architecture.
- `cbam.py`: Implementation of the Convolutional Block Attention Module.
- `main.py`: Main script for training and evaluating the model.

## Dataset
The training and evaluation are performed on the UCF-101 dataset, which is a widely recognized benchmark for action recognition.

##Internship Report
A detailed report of this internship project can be found here: [Human Activity Recognition Internship Report.pdf](Human Activity Recognition Internship Report.pdf).
