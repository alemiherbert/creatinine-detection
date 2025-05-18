import os
import argparse
from detector import CreatinineDetectionSystem

def main():
    """
    Main function for testing and making predictions with a trained model.
    """
    # Configure argument parser
    parser = argparse.ArgumentParser(description="Test a Creatinine Detection Model")
    parser.add_argument("--model", default="creatinine_model.joblib", help="Path to the trained model")
    parser.add_argument("--image", default=None, help="Path to image for prediction")
    parser.add_argument("--visualize", action="store_true", help="Visualize results on random samples")
    parser.add_argument("--dataset", default="data_interpolated", help="Dataset for visualization (only used with --visualize)")
    args = parser.parse_args()
    
    try:
        # Check if model exists
        if not os.path.exists(args.model):
            print(f"Error: Model not found at {args.model}")
            print("Please train a model first using train.py")
            return
        
        # Load the model
        print(f"Loading model from {args.model}...")
        system = CreatinineDetectionSystem(args.model)
        
        # Visualize results if requested
        if args.visualize:
            if not os.path.exists(args.dataset):
                print(f"Error: Dataset not found at {args.dataset}")
                return
            system.visualize_results(args.dataset)
        
        # Make prediction if an image is specified
        if args.image:
            if os.path.exists(args.image):
                prediction = system.predict(args.image)
                print(f"Predicted creatinine concentration: {prediction:.2f} mg/L")
            else:
                print(f"Error: Image file not found: {args.image}")
        
        # If no specific action was requested, prompt for prediction
        if not args.visualize and not args.image:
            print("\nUsing trained model for prediction")
            print("-------------------------------------")
            test_image = input("Enter the path to a test image: ")
            
            if test_image and os.path.exists(test_image):
                prediction = system.predict(test_image)
                print(f"Predicted creatinine concentration: {prediction:.2f} mg/L")
            else:
                print("Invalid image path.")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()