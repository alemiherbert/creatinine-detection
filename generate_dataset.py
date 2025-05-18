import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import random
import shutil

def generate_interpolated_dataset(input_folder="data", 
                                output_folder="data_interpolated", 
                                num_samples=100, 
                                min_concentration=10, 
                                max_concentration=100,
                                target_size=(48, 48)):
    """
    Generate an interpolated dataset from a small set of creatinine test images.
    
    Args:
        input_folder: Folder containing original images
        output_folder: Folder to save generated images
        num_samples: Number of samples to generate
        min_concentration: Minimum concentration value (mg/L)
        max_concentration: Maximum concentration value (mg/L)
        target_size: Output image size (width, height)
    """
    print(f"Loading original images from {input_folder}...")
    
    # Create output folder if it doesn't exist
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)  # Remove existing folder
    os.makedirs(output_folder)
    
    # Load all images and their concentrations
    images = []
    concentrations = []
    
    for filename in sorted(os.listdir(input_folder)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                # Extract concentration from filename
                concentration = float(filename.split('.')[0])
                
                # Load and resize image
                img_path = os.path.join(input_folder, filename)
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"Failed to load image: {img_path}")
                    continue
                
                # Resize to target size
                img = cv2.resize(img, target_size)
                
                images.append(img)
                concentrations.append(concentration)
                
                print(f"Loaded: {filename} - {concentration} mg/L")
                
            except (ValueError, AttributeError) as e:
                print(f"Skipping {filename}: {e}")
    
    if len(images) < 2:
        raise ValueError("Need at least 2 images to perform interpolation")
    
    # Sort images by concentration
    sorted_indices = np.argsort(concentrations)
    images = [images[i] for i in sorted_indices]
    concentrations = [concentrations[i] for i in sorted_indices]
    
    print(f"\nOriginal concentrations: {concentrations}")
    
    # Generate new concentration values
    if num_samples <= 1:
        raise ValueError("num_samples must be greater than 1")
    
    new_concentrations = np.linspace(min_concentration, max_concentration, num_samples)
    
    # Create interpolation functions for each pixel and channel
    height, width, channels = images[0].shape
    
    print("\nGenerating interpolated images...")
    
    # For each new concentration, generate a new image
    for new_conc in new_concentrations:
        # Find the two closest original images
        if new_conc <= concentrations[0]:
            # Extrapolation below the minimum concentration
            idx1, idx2 = 0, 1
        elif new_conc >= concentrations[-1]:
            # Extrapolation above the maximum concentration
            idx1, idx2 = len(concentrations) - 2, len(concentrations) - 1
        else:
            # Interpolation
            idx2 = np.searchsorted(concentrations, new_conc)
            idx1 = idx2 - 1
        
        # Calculate interpolation weight
        weight = (new_conc - concentrations[idx1]) / (concentrations[idx2] - concentrations[idx1])
        
        # Linear interpolation between images
        new_img = np.zeros((height, width, channels), dtype=np.float32)
        
        for c in range(channels):
            for h in range(height):
                for w in range(width):
                    pixel_val1 = float(images[idx1][h, w, c])
                    pixel_val2 = float(images[idx2][h, w, c])
                    new_pixel = pixel_val1 + weight * (pixel_val2 - pixel_val1)
                    
                    # Add small random noise to make images slightly different
                    noise = random.uniform(-2, 2)
                    new_pixel = max(0, min(255, new_pixel + noise))
                    new_img[h, w, c] = new_pixel
        
        # Convert to uint8
        new_img = new_img.astype(np.uint8)
        
        # Save the new image
        output_filename = f"{new_conc:.1f}.jpg"
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, new_img)
        
        print(f"Generated: {output_filename}")
    
    print(f"\nGenerated {len(new_concentrations)} interpolated images in {output_folder}")
    
    # Visualize some examples
    visualize_results(input_folder, output_folder)

def visualize_results(input_folder, output_folder):
    """
    Visualize some original and generated images
    """
    # Get a few original and generated images
    original_files = sorted([f for f in os.listdir(input_folder) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    generated_files = sorted([f for f in os.listdir(output_folder) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    # Sample some generated images
    sample_size = min(5, len(generated_files))
    step = len(generated_files) // sample_size
    sampled_generated = [generated_files[i] for i in range(0, len(generated_files), step)][:sample_size]
    
    plt.figure(figsize=(15, 8))
    
    # Plot original images
    for i, filename in enumerate(original_files):
        img = cv2.imread(os.path.join(input_folder, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.subplot(2, max(len(original_files), len(sampled_generated)), i + 1)
        plt.imshow(img)
        plt.title(f"Original: {filename}")
        plt.axis('off')
    
    # Plot generated images
    for i, filename in enumerate(sampled_generated):
        img = cv2.imread(os.path.join(output_folder, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.subplot(2, max(len(original_files), len(sampled_generated)), 
                  len(original_files) + i + 1)
        plt.imshow(img)
        plt.title(f"Generated: {filename}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_comparison.png')
    plt.show()

if __name__ == "__main__":
    # Generate interpolated dataset
    generate_interpolated_dataset(
        input_folder="data",
        output_folder="data_interpolated",
        num_samples=100,
        min_concentration=10,
        max_concentration=100,
        target_size=(48, 48)
    )