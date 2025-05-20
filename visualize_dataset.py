import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
import argparse
from matplotlib.gridspec import GridSpec

def visualize_color_histograms(dataset_path, img_ext='.jpg', samples_per_row=3, num_samples=None):
    """
    Visualize color histograms for creatinine test images.
    
    Args:
        dataset_path: Path to the folder containing images
        img_ext: Image file extension to look for
        samples_per_row: Number of samples to display per row
        num_samples: Maximum number of samples to display (None for all)
    """
    # Get all image files
    image_files = []
    for filename in os.listdir(dataset_path):
        if filename.lower().endswith(img_ext):
            try:
                # Extract concentration from filename
                concentration = float(re.search(r'(\d+(\.\d+)?)', filename).group(1))
                image_files.append((filename, concentration))
            except (ValueError, AttributeError) as e:
                print(f"Skipping {filename}: {e}")
    
    # Sort by concentration
    image_files.sort(key=lambda x: x[1])
    
    # Limit samples if specified
    if num_samples is not None:
        if len(image_files) > num_samples:
            # Sample evenly from the range
            step = max(1, len(image_files) // num_samples)
            image_files = image_files[::step][:num_samples]
    
    num_images = len(image_files)
    if num_images == 0:
        print("No valid images found.")
        return
    
    # Calculate grid dimensions
    num_rows = (num_images + samples_per_row - 1) // samples_per_row
    
    # Create figure
    fig = plt.figure(figsize=(16, 4 * num_rows))
    gs = GridSpec(num_rows, samples_per_row * 2)  # Each sample needs 2 columns (image + histograms)
    
    # Color spaces to analyze
    color_spaces = [
        {"name": "BGR", "conversion": None, "channels": ['Blue', 'Green', 'Red'], "colors": ['b', 'g', 'r']},
        {"name": "HSV", "conversion": cv2.COLOR_BGR2HSV, "channels": ['Hue', 'Saturation', 'Value'], 
         "colors": ['purple', 'orange', 'black']},
        {"name": "LAB", "conversion": cv2.COLOR_BGR2LAB, "channels": ['L', 'A', 'B'], 
         "colors": ['black', 'green', 'blue']}
    ]
    
    # Process each image
    for i, (filename, concentration) in enumerate(image_files):
        row = i // samples_per_row
        col = i % samples_per_row
        
        # Load image
        img_path = os.path.join(dataset_path, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
        
        # Display original image
        ax_img = fig.add_subplot(gs[row, col*2])
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax_img.imshow(rgb_img)
        ax_img.set_title(f"{concentration} mg/L", size=12)
        ax_img.axis('off')
        
        # Display histograms
        ax_hist = fig.add_subplot(gs[row, col*2+1])
        
        # For each color space
        for cs_idx, color_space in enumerate(color_spaces):
            # Convert image if needed
            if color_space["conversion"] is not None:
                converted_img = cv2.cvtColor(img, color_space["conversion"])
            else:
                converted_img = img
                
            # Plot histogram for each channel
            for channel_idx, (channel_name, color) in enumerate(zip(color_space["channels"], color_space["colors"])):
                hist = cv2.calcHist([converted_img], [channel_idx], None, [256], [0, 256])
                # Normalize for better visualization
                hist = hist / hist.max() * 0.9  # Scale to 0.9 max height
                # Offset each color space for better visualization
                x_offset = cs_idx * 256
                x_values = np.arange(256) + x_offset
                ax_hist.plot(x_values, hist, color=color, label=f"{color_space['name']}-{channel_name}")
                
        if i == 0:  # Only show legend on first plot
            ax_hist.legend(loc='upper right', fontsize=8)
            
        ax_hist.set_xlim(0, 256 * len(color_spaces))
        ax_hist.set_xticks([128 + 256*i for i in range(len(color_spaces))])
        ax_hist.set_xticklabels([cs["name"] for cs in color_spaces])
        ax_hist.set_title("Color Histograms", size=10)
        ax_hist.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('color_histograms.png', dpi=150, bbox_inches='tight')
    plt.show()

def visualize_feature_trends(dataset_path, img_ext='.jpg'):
    """
    Visualize how color features change with concentration levels.
    
    Args:
        dataset_path: Path to the folder containing images
        img_ext: Image file extension to look for
    """
    # Get all image files and their concentrations
    data = []
    for filename in os.listdir(dataset_path):
        if filename.lower().endswith(img_ext):
            try:
                # Extract concentration from filename
                concentration = float(re.search(r'(\d+(\.\d+)?)', filename).group(1))
                
                # Load and process image
                img_path = os.path.join(dataset_path, filename)
                img = cv2.imread(img_path)
                
                if img is None:
                    continue
                
                # Store concentration and image
                data.append((concentration, img))
            except (ValueError, AttributeError) as e:
                print(f"Skipping {filename}: {e}")
    
    # Sort by concentration
    data.sort(key=lambda x: x[0])
    
    if not data:
        print("No valid images found.")
        return
    
    # Extract concentrations and prepare features arrays
    concentrations = [item[0] for item in data]
    
    # Color spaces to analyze
    color_spaces = [
        {"name": "BGR", "conversion": None, "channels": ['Blue', 'Green', 'Red']},
        {"name": "HSV", "conversion": cv2.COLOR_BGR2HSV, "channels": ['Hue', 'Saturation', 'Value']},
        {"name": "LAB", "conversion": cv2.COLOR_BGR2LAB, "channels": ['L', 'A', 'B']}
    ]
    
    # Create figure for color means
    plt.figure(figsize=(12, 15))
    
    # Plot each color space in a separate subplot
    for cs_idx, color_space in enumerate(color_spaces):
        plt.subplot(3, 1, cs_idx + 1)
        
        channel_means = []
        channel_stds = []
        
        # Calculate means and stds for each channel
        for channel_idx in range(3):
            means = []
            stds = []
            
            for _, img in data:
                # Convert image if needed
                if color_space["conversion"] is not None:
                    converted_img = cv2.cvtColor(img, color_space["conversion"])
                else:
                    converted_img = img
                
                # Extract channel
                channel = converted_img[:, :, channel_idx]
                means.append(np.mean(channel))
                stds.append(np.std(channel))
            
            channel_means.append(means)
            channel_stds.append(stds)
        
        # Plot means with error bars (std)
        for channel_idx, channel_name in enumerate(color_space["channels"]):
            plt.errorbar(concentrations, channel_means[channel_idx], 
                       yerr=channel_stds[channel_idx], 
                       label=channel_name,
                       marker='o', capsize=4)
        
        plt.title(f"{color_space['name']} Channel Means vs Concentration")
        plt.xlabel("Concentration (mg/L)")
        plt.ylabel("Channel Mean Value")
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('color_trends.png', dpi=150)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize color histograms from training data")
    parser.add_argument("--dataset", default="training_data", help="Path to dataset folder")
    parser.add_argument("--samples", type=int, default=9, help="Number of samples to display")
    parser.add_argument("--columns", type=int, default=3, help="Number of samples per row")
    parser.add_argument("--trends", action="store_true", help="Visualize feature trends")
    args = parser.parse_args()
    
    try:
        # Check if dataset exists
        if not os.path.exists(args.dataset):
            print(f"Error: Dataset path '{args.dataset}' not found")
            return
        
        # Visualize histograms
        print(f"Generating color histograms for {args.samples} samples...")
        visualize_color_histograms(args.dataset, samples_per_row=args.columns, num_samples=args.samples)
        
        # Visualize trends if requested
        if args.trends:
            print("Generating color trend analysis...")
            visualize_feature_trends(args.dataset)
            
        print("Visualization complete. Images saved to color_histograms.png and color_trends.png")
            
    except Exception as e:
        print(f"Error during visualization: {e}")

if __name__ == "__main__":
    main()