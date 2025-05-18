import os
import argparse
from detector import CreatinineDetectionSystem

def main():
    """
    Main function for training the creatinine detection model.
    """
    # Configure argument parser
    parser = argparse.ArgumentParser(description="Train a Creatinine Detection Model")
    parser.add_argument("--dataset", default="training_data", help="Path to dataset folder")
    parser.add_argument("--model", default="creatinine_model.joblib", help="Path to save the trained model")
    parser.add_argument("--visualize", action="store_true", help="Visualize training results")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data to use for testing")
    args = parser.parse_args()
    
    # Define paths
    dataset_path = args.dataset
    model_path = args.model
    
    try:
        # Create and train the system
        print(f"Training new model using dataset from {dataset_path}...")
        system = CreatinineDetectionSystem()
        metrics = system.train(dataset_path, test_size=args.test_size)
        
        # Save the model
        system.save_model(model_path)
        print(f"Model successfully trained and saved to {model_path}")
        print(f"Model performance metrics:")
        print(f"  RMSE: {metrics['rmse']:.2f} mg/L")
        print(f"  R²: {metrics['r2']:.2f}")
        
        # Visualize results if requested
        if args.visualize:
            system.visualize_results(dataset_path)
    
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    main()