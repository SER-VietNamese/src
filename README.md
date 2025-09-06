# Speech Emotion Recognition (SER) Models for Vietnamese

This repository contains five different deep learning models for Speech Emotion Recognition (SER) on Vietnamese audio data. The models are implemented using TensorFlow/Keras and leverage various audio feature extraction techniques.

## Table of Contents

- [Speech Emotion Recognition (SER) Models for Vietnamese](#speech-emotion-recognition-ser-models-for-vietnamese)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Dataset](#dataset)
  - [Feature Extraction](#feature-extraction)
  - [Models Implemented](#models-implemented)
  - [Results](#results)
  - [Setup and Usage](#setup-and-usage)

## Project Overview

Speech Emotion Recognition (SER) aims to identify human emotions from speech. This project explores different neural network architectures to classify emotions (Angry, Happy, Anxiety, Neutral, Sad) from Vietnamese spoken words. Each model notebook demonstrates data loading, feature extraction, model building, training, evaluation, and prediction.

## Dataset

The models are trained and evaluated on a Vietnamese speech emotion dataset. The dataset is structured into folders, with each folder representing a specific emotion:
- `vesc/`
  - `Angry/`
  - `Happy/`
  - `Anxiety/`
  - `Neutral/`
  - `Sad/`
- `vesc_test/` (Test set)
  - `Angry/`
  - `Happy/`
  - `Anxiety/`
  - `Neutral/`
  - `Sad/`

The `*.wav` audio files within these folders are used for training and testing.

## Feature Extraction

All models use a comprehensive set of audio features extracted using `librosa`:
- **MFCCs (Mel-frequency cepstral coefficients)**: Captures the spectral envelope of the sound.
- **Delta and Delta-Delta MFCCs**: Represents the first and second derivatives of MFCCs, providing information about changes in the vocal tract.
- **Pitch Features**:
  - **Mean Pitch**: The average fundamental frequency.
  - **Jitter**: Variation in pitch.
- And many other features as detailed in the notebooks.

## Models Implemented

Below is a list of the models implemented in this project. Each model has its own notebook in the `src/` directory with implementation details and results.

1.  **OrionNet Model**:
    -   **Notebook**: [`src/ser-vesc-orionnet.ipynb`](ser-vesc-orionnet.ipynb)
    -   **Description**: Uses a Convolutional Neural Network (CNN) for emotion classification.

2.  **Transformer Model**:
    -   **Notebook**: [`src/ser-vesc-transformer.ipynb`](ser-vesc-trasformer.ipynb)
    -   **Description**: Leverages the Transformer architecture to capture relationships in speech data.

3.  **CNN + BiLSTM + Attention Model**:
    -   **Notebook**: [`src/ser-vesc-cnn-bilstm-att.ipynb`](ser-vesc-cnn-bilstm-att.ipynb)
    -   **Description**: A hybrid model that uses a CNN for feature extraction, a BiLSTM to model time series, and an attention mechanism to focus on important parts of the speech.

4.  **ESERNet Model**:
    -   **Notebook**: [`src/ser-vesc-esernet.ipynb`](ser-vesc-esernet.ipynb)
    -   **Description**: A newly added model, likely an efficient architecture for speech emotion recognition.

5.  **STACN Model**:
    -   **Notebook**: [`src/ser-vesc-stacn.ipynb`](ser-vesc-stacn.ipynb)
    -   **Description**: Another advanced model added, which may use state-of-the-art techniques to improve accuracy.

## Results

Detailed results for each model, including accuracy, confusion matrices, and other metrics, can be found in their respective notebooks.

## Setup and Usage

To run these notebooks, you will need to install the necessary libraries. Detailed requirements are listed within each notebook file. In general, you will need:
-   TensorFlow
-   Librosa
-   Scikit-learn
-   And other libraries used in the notebooks.