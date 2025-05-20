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
    parser.add_argument("--dilution", type=float, default=50.0, help="Dilution factor (default: 50x)")
    parser.add_argument("--upload", action="store_true", help="Upload image from Google Colab")
    args = parser.parse_args()
    
    try:
        # Check if running in Google Colab for upload functionality
        in_colab = 'google.colab' in sys.modules
        
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
            
        # Handle file upload in Google Colab if requested
        if in_colab and args.upload:
            try:
                from google.colab import files
                print("Please upload an image for prediction...")
                uploaded = files.upload()
                
                if uploaded:
                    # Get the first uploaded file
                    filename = next(iter(uploaded))
                    image_path = f"test_{filename}"
                    
                    # Save the uploaded file
                    with open(image_path, 'wb') as f:
                        f.write(uploaded[filename])
                    
                    print(f"Image saved as {image_path}")
                    args.image = image_path
                else:
                    print("No image uploaded.")
            except ImportError:
                print("Warning: Google Colab modules not available. Skipping upload.")
            except Exception as upload_error:
                print(f"Error during file upload: {upload_error}")
        
        # Make prediction if an image is specified
        if args.image:
            if os.path.exists(args.image):
                # Get diluted prediction
                diluted_prediction = system.predict(args.image)
                
                # Calculate original concentration based on dilution factor
                original_prediction = diluted_prediction * args.dilution
                
                # Display results
                print(f"Predicted creatinine concentration (diluted sample): {diluted_prediction:.2f} mg/L")
                print(f"Predicted creatinine concentration (original urine): {original_prediction:.2f} mg/L")
                print(f"(Based on a {args.dilution}x dilution factor)")
                
                # If in Colab, display the image
                if in_colab:
                    try:
                        from IPython.display import Image, display
                        display(Image(args.image))
                    except ImportError:
                        pass
            else:
                print(f"Error: Image file not found: {args.image}")
        
        # If no specific action was requested, prompt for prediction
        if not args.visualize and not args.image:
            print("\nUsing trained model for prediction")
            print("-------------------------------------")
            
            # For Colab, prompt for upload
            if in_colab:
                try:
                    from google.colab import files
                    print("Please upload an image for prediction...")
                    uploaded = files.upload()
                    
                    if uploaded:
                        # Get the first uploaded file
                        filename = next(iter(uploaded))
                        test_image = f"test_{filename}"
                        
                        # Save the uploaded file
                        with open(test_image, 'wb') as f:
                            f.write(uploaded[filename])
                        
                        print(f"Image saved as {test_image}")
                    else:
                        print("No image uploaded.")
                        return
                except ImportError:
                    test_image = input("Enter the path to a test image: ")
            else:
                test_image = input("Enter the path to a test image: ")
            
            if test_image and os.path.exists(test_image):
                # Get diluted prediction
                diluted_prediction = system.predict(test_image)
                
                # Calculate original concentration based on dilution factor
                original_prediction = diluted_prediction * args.dilution
                
                # Display results
                print(f"Predicted creatinine concentration (diluted sample): {diluted_prediction:.2f} mg/L")
                print(f"Predicted creatinine concentration (original urine): {original_prediction:.2f} mg/L")
                print(f"(Based on a {args.dilution}x dilution factor)")
            else:
                print("Invalid image path.")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    main()