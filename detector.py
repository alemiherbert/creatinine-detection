import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import re

class CreatinineDetectionSystem:
    def __init__(self, model_path=None):
        """
        Initialize the creatinine detection system.
        
        Args:
            model_path: Path to a pre-trained model. If None, a new model will be trained.
        """
        self.model = None
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
    
    def extract_features(self, image):
        """
        Extract color features from an image.
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            feature_vector: Extracted features
        """
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Calculate mean and std for each channel in each color space
        features = []
        
        # BGR features
        for i, color in enumerate(['blue', 'green', 'red']):
            channel = image[:,:,i]
            features.extend([
                np.mean(channel),  # Mean
                np.std(channel),   # Standard deviation
                np.percentile(channel, 25),  # 25th percentile
                np.percentile(channel, 75)   # 75th percentile
            ])
        
        # HSV features
        for i, color in enumerate(['hue', 'saturation', 'value']):
            channel = hsv[:,:,i]
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.percentile(channel, 25),
                np.percentile(channel, 75)
            ])
        
        # LAB features
        for i, color in enumerate(['l', 'a', 'b']):
            channel = lab[:,:,i]
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.percentile(channel, 25),
                np.percentile(channel, 75)
            ])
            
        # Add histogram features
        for i in range(3):  # For each channel in BGR
            hist = cv2.calcHist([image], [i], None, [10], [0, 256])
            hist = hist.flatten() / hist.sum()  # Normalize
            features.extend(hist)
            
        return np.array(features)
    
    def load_dataset(self, dataset_path, img_ext='.jpg'):
        """
        Load the dataset of images and extract their labels from filenames.
        
        Args:
            dataset_path: Path to the folder containing images
            img_ext: Image file extension to look for
            
        Returns:
            X: Features extracted from images
            y: Creatinine concentration labels
        """
        X = []
        y = []
        
        for filename in os.listdir(dataset_path):
            if filename.endswith(img_ext):
                # Extract concentration from filename (e.g., "50.jpg" → 50 mg/L)
                try:
                    concentration = float(re.search(r'(\d+(\.\d+)?)', filename).group(1))
                    
                    # Load and process the image
                    img_path = os.path.join(dataset_path, filename)
                    image = cv2.imread(img_path)
                    
                    if image is None:
                        print(f"Failed to load image: {img_path}")
                        continue
                    
                    # Extract features
                    features = self.extract_features(image)
                    
                    X.append(features)
                    y.append(concentration)
                except (ValueError, AttributeError) as e:
                    print(f"Skipping {filename}: {e}")
        
        return np.array(X), np.array(y)
    
    def train(self, dataset_path, test_size=0.2, random_state=42):
        """
        Train the model on the dataset.
        
        Args:
            dataset_path: Path to the folder containing images
            test_size: Fraction of the dataset to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            test_metrics: Dictionary containing test metrics
        """
        # Load dataset
        print("Loading dataset...")
        X, y = self.load_dataset(dataset_path)
        
        if len(X) == 0:
            raise ValueError("No valid images found in the dataset.")
        
        print(f"Dataset loaded: {len(X)} samples")
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train Random Forest Regressor
        print("Training model...")
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        test_metrics = {
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
        
        print(f"Model trained. Test RMSE: {rmse:.2f}, R²: {r2:.2f}")
        
        return test_metrics
    
    def save_model(self, model_path):
        """
        Save the trained model to disk.
        
        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")
    
    def predict(self, image_path):
        """
        Predict the creatinine concentration from a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            concentration: Predicted creatinine concentration in mg/L
        """
        if self.model is None:
            raise ValueError("No model available. Train or load a model first.")
        
        # Load and process the image
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Extract features
        features = self.extract_features(image)
        
        # Make prediction
        concentration = self.model.predict([features])[0]
        
        return concentration
    
    def visualize_results(self, dataset_path, n_samples=10):
        """
        Visualize the model's predictions on random samples from the dataset.
        
        Args:
            dataset_path: Path to the folder containing images
            n_samples: Number of samples to visualize
        """
        if self.model is None:
            raise ValueError("No model available. Train or load a model first.")
        
        # Load dataset
        X, y = self.load_dataset(dataset_path)
        
        # Get random indices
        indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
        
        # Make predictions
        X_sample = X[indices]
        y_true = y[indices]
        y_pred = self.model.predict(X_sample)
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--')
        plt.xlabel('True Concentration (mg/L)')
        plt.ylabel('Predicted Concentration (mg/L)')
        plt.title('True vs Predicted Creatinine Concentration')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        for i, (true, pred) in enumerate(zip(y_true, y_pred)):
            print(f"Sample {i+1}: True = {true:.1f} mg/L, Predicted = {pred:.1f} mg/L, Error = {(pred-true):.1f} mg/L")