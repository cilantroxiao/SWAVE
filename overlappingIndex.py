import tkinter as tk
from tkinter import filedialog
import pandas as pd
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime

def output_1(image):
    # Detect edges using Canny edge detection
    edges = cv2.Canny(image, 50, 150)
    
    # Find contours of the detected edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest rectangular contour (likely the graph border)
    graph_contour = None
    max_area = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > max_area and w > 0.5 * image.shape[1] and h > 0.5 * image.shape[0]:
            max_area = area
            graph_contour = (x, y, w, h)
    
    if graph_contour:
        x, y, w, h = graph_contour
        return image[y:y+h, x:x+w]  # Crop to the graph region
    else:
        return image  # Return original image if no graph border is detected

def calculate_overlap_index(image_paths):
    images = [cv2.imread(img_path, cv2.IMREAD_COLOR) for img_path in image_paths if os.path.exists(img_path)]  # Load original colored images
    if not images:
        print(f"No images found for: {image_paths}")
        return None, []
    
    # Resize images to the smallest dimensions
    min_height = min(img.shape[0] for img in images)
    min_width = min(img.shape[1] for img in images)
    resized_images = [cv2.resize(img, (min_width, min_height)) for img in images]

    # Apply graph-region cropping to all images
    graph_cropped_images = [output_1(img) for img in resized_images]

    # Resize graph-cropped images to the smallest dimensions for alignment
    min_height_cropped = min(img.shape[0] for img in graph_cropped_images)
    min_width_cropped = min(img.shape[1] for img in graph_cropped_images)
    aligned_graph_images = [cv2.resize(img, (min_width_cropped, min_height_cropped)) for img in graph_cropped_images]

    graph_composite_image = np.sum([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in aligned_graph_images], axis=0).astype(np.float32)  # Convert to float32

    # Normalize graph composite image for visualization
    graph_composite_normalized = cv2.normalize(
        graph_composite_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    return graph_composite_normalized, resized_images  # Return original resized images instead of gray-cropped images

def save_group_composite(output_path, group_index, composite_image, individual_images):
    composite_image_filename = os.path.join(output_path, f"group_{group_index}_composite_with_individuals.png")

    # Create a visual layout: individual images on the left, composite on the right
    num_individuals = len(individual_images)
    fig, axs = plt.subplots(1, 2, figsize=(15, 10), gridspec_kw={'width_ratios': [2, 1]})

    # Left: Grid of individual input images
    num_rows = (num_individuals + 1) // 2
    fig_individuals, axs_individuals = plt.subplots(num_rows, 2, figsize=(10, 10))
    axs_individuals = axs_individuals.flatten()

    for i, img in enumerate(individual_images):
        axs_individuals[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert to RGB for display
        axs_individuals[i].axis('off')
        axs_individuals[i].set_title(f"Wave {i + 1}")

    for j in range(len(individual_images), len(axs_individuals)):
        axs_individuals[j].axis('off')  # Hide empty subplots

    fig_individuals.suptitle("Original Wave Images")
    fig_individuals.tight_layout()
    individual_grid_path = os.path.join(output_path, f"group_{group_index}_individuals_grid.png")
    plt.savefig(individual_grid_path)
    plt.close(fig_individuals)

    # Right: Composite image
    axs[0].imshow(plt.imread(individual_grid_path))  # Load saved grid image for display
    axs[0].axis('off')
    axs[0].set_title("Original Input Waves")

    axs[1].imshow(composite_image, cmap='gray')
    axs[1].axis('off')
    axs[1].set_title("Group Composite")

    plt.tight_layout()
    plt.savefig(composite_image_filename)
    plt.close()

    print(f"Saved composite with individual graphs for Group {group_index} at: {composite_image_filename}")

def open_directory():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    selected_directory = filedialog.askdirectory(title="Select Mouse Directory")
    
    if selected_directory:
        # Navigate to the `stage05_wave_characterization/time_stamp` folder
        stage05_path = os.path.join(selected_directory, 'stage05_wave_characterization', 'time_stamp')
        csv_file_path = os.path.join(stage05_path, 'wavefronts_time_stamp.csv')
        label_planar_path = os.path.join(selected_directory, 'stage05_wave_characterization', 'label_planar')
        base_output_dir = os.path.join(selected_directory, 'outputs_overlapping_index')

        # Create base output directory if it doesn't exist
        if not os.path.exists(base_output_dir):
            os.makedirs(base_output_dir)
            print(f"Created base output directory: {base_output_dir}")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        mouse_name = os.path.basename(selected_directory)
        output_path = os.path.join(base_output_dir, f"{timestamp}_{mouse_name}")

        # Create directory for all groups combined
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(f"Created output directory for all group composites: {output_path}")
        
        if os.path.exists(csv_file_path):
            # Read the CSV file
            df = pd.read_csv(csv_file_path)
            print("CSV file loaded successfully.")
            
            wave_groups = []
            current_group = [(df.loc[0, 'wavefronts_id'], df.loc[0, 'time_stamp'])]  # Start the first group
            
            for i in range(1, len(df)):
                current_id = df.loc[i, 'wavefronts_id']
                current_time = df.loc[i, 'time_stamp']
                previous_time = df.loc[i - 1, 'time_stamp']
                
                if abs(current_time - previous_time) <= 0.5:
                    current_group.append((current_id, current_time))
                else:
                    wave_groups.append(current_group)
                    current_group = [(current_id, current_time)]
            
            # Append the last group
            wave_groups.append(current_group)
            
            # Calculate the overlap index for each group and save in the same directory
            if wave_groups:
                for idx, group in enumerate(wave_groups):
                    if len(group) > 1:  # Only process groups with more than one wavefront
                        image_paths = [os.path.join(label_planar_path, f"wave_{wave_id}.png") for wave_id, _ in group]
                        print(f"\nProcessing Group {idx + 1}:")
                        print([f"(ID: {wave_id}, Time: {timestamp}s)" for wave_id, timestamp in group])
                        print("Calculating overlap index...")
                        composite_image, individual_images = calculate_overlap_index(image_paths)
                        save_group_composite(output_path, idx + 1, composite_image, individual_images)
            else:
                print("No grouped wavefronts found.")
        else:
            print(f"CSV file not found at {os.path.abspath(csv_file_path)}.")
    else:
        print("No directory selected.")

if __name__ == "__main__":
    open_directory()