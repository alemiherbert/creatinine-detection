# Creatinine Detection System Documentation

## Overview
The Creatinine Detection System is a machine learning-based tool for analyzing images of creatinine test samples and accurately predicting the concentration level. The system uses computer vision techniques to extract image features and a Random Forest Regressor model to predict creatinine concentration values from test images.

## System Components

### 1. `detector.py`
Core module containing the `CreatinineDetectionSystem` class with all image processing and machine learning functionality.

### 2. `train.py`
Script for training a new detection model using a labeled dataset.

### 3. `test.py`
Script for making predictions using a previously trained model.

### 4. `generate_dataset.py` (Optional)
Utility for generating interpolated test images from a small original dataset.

## Installation

### Requirements
- Python 3.7+
- OpenCV
- scikit-learn
- NumPy
- Matplotlib
- joblib

Install dependencies:
```bash
pip install opencv-python scikit-learn numpy matplotlib joblib
```
### Usage
1. Data Preparation
Place your creatinine test images in a directory (default: training_data)
Images should be named with their concentration values (e.g., 50.jpg for 50 mg/L)
For best results, use 48x48 pixel images
2. Training a Model
### Options:

- `--dataset`: Path to the folder containing training images (default: "training_data")
- `--model`: Path to save the trained model (default: "creatinine_model.joblib")
- `--visualize`: Enable visualization of training results
- `--test-size`: Fraction of data to use for testing (default: 0.2)
3. Making Predictions
```(bash)
python test.py --image path/to/test_image.jpg
```
Options:

- `--model`: Path to the trained model file (default: "creatinine_model.joblib")
- `--image`: Path to the image for prediction
- `--visualize`: Visualize results on random samples from the dataset
- `--dataset`: Dataset path for visualization (only used with --visualize)
### Technical Details
#### Feature Extraction
The system extracts the following features from each image:

- Color statistics (mean, standard deviation, percentiles) in BGR, HSV, and LAB color spaces
- Color histograms for each BGR channel
#### Model
- Algorithm: Random Forest Regressor
#### Parameters:
- n_estimators: 100
- max_depth: 20
- min_samples_split: 2
- min_samples_leaf: 2
#### Performance Metrics
- RMSE (Root Mean Square Error): Measures prediction accuracy in mg/L
- R² (Coefficient of determination): Measures the proportion of variance explained by the model