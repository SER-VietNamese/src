# Speech Emotion Recognition (SER) Models for Vietnamese

This repository contains three different deep learning models for Speech Emotion Recognition (SER) on Vietnamese audio data. The models are implemented using TensorFlow/Keras and leverage various audio feature extraction techniques.

## Table of Contents

- [Speech Emotion Recognition (SER) Models for Vietnamese](#speech-emotion-recognition-ser-models-for-vietnamese)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Dataset](#dataset)
  - [Feature Extraction](#feature-extraction)
  - [Models Implemented](#models-implemented)
    - [1. CNN Model](#1-cnn-model)
    - [2. Transformer Model](#2-transformer-model)
    - [3. CNN + BiLSTM + Attention Model](#3-cnn--bilstm--attention-model)
  - [Results](#results)
  - [Setup and Usage](#setup-and-usage)

## Project Overview

Speech Emotion Recognition (SER) aims to identify human emotions from speech. This project explores different neural network architectures to classify emotions (angry, happiness, anxiety, neutral, sadness) from Vietnamese spoken words. Each model notebook demonstrates data loading, feature extraction, model building, training, evaluation, and prediction.

## Dataset

The models are trained and evaluated on a Vietnamese speech emotion dataset.
The dataset is structured into folders, with each folder representing a specific emotion:
- `emotion_vi_train/`
  - `angry/`
  - `happiness/`
  - `anxiety/`
  - `neutral/`
  - `sadness/`
- `emotion_vi_test_n_o/` (Test set)
  - `angry/`
  - `happiness/`
  - `anxiety/`
  - `neutral/`
  - `sadness/`

The `*.wav` audio files within these folders are used for training and testing.

## Feature Extraction

All models utilize a comprehensive set of audio features extracted using `librosa`:
- **MFCCs (Mel-frequency cepstral coefficients)**: Captures the spectral envelope of a sound.
- **Delta and Delta-Delta MFCCs**: Represent the first and second derivatives of MFCCs, providing information about changes in the vocal tract.
- **Pitch Features**:
    - **Mean Pitch**: Average fundamental frequency.
    - **Jitter**: Variation in pitch period.
    - **Pitch Slope**: Trend of pitch change.
    - A global mean pitch, jitter, and pitch slope are calculated from the entire training dataset for normalization.
- **Energy Features**:
    - **RMS (Root Mean Square) Energy**: Measures the overall loudness.
    - **Mean Energy**: Average energy of the audio.
    - **Shimmer**: Variation in amplitude.
- **Zero-Crossing Rate (ZCR)**: Indicates the rate at which the signal changes sign, useful for differentiating voiced and unvoiced speech.
- **Chroma Features**: Represents the 12 different pitch classes, robust to timbral changes.
- **Spectral Features**:
    - **Spectral Centroid**: The "center of mass" of the spectrum, indicating brightness.
    - **Spectral Rolloff**: The frequency below which a specified percentage of the total spectral energy, e.g., 85%, is contained.
    - **Spectral Contrast**: Measures the energy difference between peaks and valleys in the spectrum, reflecting the clarity of harmonic peaks.
- **Temporal Features**:
    - **Speech Rate**: Number of voiced frames per second.
    - **Pauses**: Number of silent segments.

**Data Augmentation**: To improve model robustness and prevent overfitting, the following augmentation techniques are applied during training:
- Adding noise
- Pitch shifting
- Time stretching
- Time shifting

Features are normalized using `StandardScaler` to ensure consistent input scales across models.

## Models Implemented

This repository showcases three distinct deep learning architectures. Each model's notebook (`.ipynb` file) contains the complete code for data preparation, model definition, training, and evaluation.

### 1. CNN Model
- **Filename**: `speechemotionvesc.ipynb`
- **Architecture**: A 2D Convolutional Neural Network (CNN) designed to process the 2D audio feature maps (features x time frames). It includes multiple `Conv2D`, `BatchNormalization`, `MaxPooling2D`, and `Dropout` layers followed by `Flatten` and `Dense` layers for classification. L2 regularization is applied to combat overfitting.

### 2. Transformer Model
- **Filename**: `speechrecognizetrasformer.ipynb`
- **Architecture**: This model utilizes Transformer blocks, known for their effectiveness in sequence modeling. The audio features are transformed into a sequence-like input for the Transformer. It incorporates positional encoding, multiple `TransformerBlock` layers (with `MultiHeadAttention`, `Dense` feed-forward networks, `LayerNormalization`, and `Dropout`), followed by a `TemporalAttention` layer to aggregate temporal information, and finally `Dense` layers for classification.

### 3. CNN + BiLSTM + Attention Model
- **Filename**: `speechemotioncnnbilstmattention.ipynb`
- **Architecture**: This hybrid model combines the strengths of CNNs, Bidirectional LSTMs (BiLSTM), and an Attention mechanism.
    - **CNN layers**: Extract local features from the audio spectrograms, similar to the pure CNN model.
    - **Reshape layer**: Transforms the 2D CNN output into a 3D sequence (batch_size, time_steps, features) suitable for recurrent layers.
    - **BiLSTM layer**: Processes the sequential features from both forward and backward directions, capturing long-range dependencies.
    - **Attention layer**: Focuses on the most relevant parts of the sequence output from the BiLSTM, allowing the model to weigh different time steps differently.
    - **Dense layers**: For the final emotion classification.

## Results

Each notebook provides detailed evaluation metrics, including:
- **Confusion Matrix**: Visualizes the performance of the classification model on the test set, showing true positive, true negative, false positive, and false negative predictions for each emotion.
- **Classification Report**: Provides precision, recall, and F1-score for each emotion class, as well as macro and weighted averages.
- **Loss History Plot**: Shows the training and validation loss over epochs, helping to monitor for overfitting.

Here's a summary of the performance metrics from the **test set** for each model:

| Model                     | Accuracy | Precision (Weighted Avg.) | Recall (Weighted Avg.) | F1-Score (Weighted Avg.) |
|---------------------------|--------------------------|-------------------------------------------|----------------------------------------|------------------------------------------|
| CNN                       | 0.9881         | 0.99                            | 0.99                         | 0.99                           |
| Transformer               | 0.9390         | 0.91                            | 0.90                         | 0.90                           |
| CNN + BiLSTM + Attention  | 0.9702         | 0.88                            | 0.87                         | 0.87                           |

*Note: The results are based on the specific dataset and training parameters used in the notebooks. Performance may vary with different data or hyperparameter tuning.*

## Setup and Usage

To run these notebooks, you'll need a Python environment with the following libraries:

```bash
pip install numpy librosa tensorflow scikit-learn matplotlib seaborn