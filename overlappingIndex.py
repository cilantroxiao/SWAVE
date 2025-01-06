import tkinter as tk
from tkinter import filedialog
import pandas as pd
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def calculate_overlap_index(image_paths, output_path, group_index):
    images = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in image_paths if os.path.exists(img_path)]
    if not images:
        print(f"No images found for: {image_paths}")
        return None
    
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

    graph_composite_image = np.sum(aligned_graph_images, axis=0).astype(np.float32)  # Convert to float32


    # Normalize graph composite image for visualization
    graph_composite_normalized = cv2.normalize(
        graph_composite_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    # Save the composite image
    composite_image_filename = os.path.join(output_path, f"wave_group_{group_index}_composite.png")
    cv2.imwrite(composite_image_filename, graph_composite_normalized)
    print(f"Saved composite image for Wave Group {group_index} at: {composite_image_filename}")

    # Display graph-cropped composite and completeness index
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Graph-Cropped Composite Image")
    plt.imshow(graph_composite_normalized, cmap='gray')
    plt.axis('off')

    return graph_composite_normalized

def open_directory():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    selected_directory = filedialog.askdirectory(title="Select Mouse Directory")
    
    if selected_directory:
        # Navigate to the `stage05_wave_characterization/time_stamp` folder
        stage05_path = os.path.join(selected_directory, 'stage05_wave_characterization', 'time_stamp')
        csv_file_path = os.path.join(stage05_path, 'wavefronts_time_stamp.csv')
        label_planar_path = os.path.join(selected_directory, 'stage05_wave_characterization', 'label_planar')
        output_dir = os.path.join(selected_directory, 'outputs_overlapping_index')

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        
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
            
            # Display the grouped waves and calculate the overlap index
            if wave_groups:
                for idx, group in enumerate(wave_groups):
                    if len(group) > 1:  # Only display groups with more than one wavefront
                        image_paths = [os.path.join(label_planar_path, f"wave_{wave_id}.png") for wave_id, _ in group]
                        print(f"\nWave Group {idx + 1}:")
                        print([f"(ID: {wave_id}, Time: {timestamp}s)" for wave_id, timestamp in group])
                        print("Calculating overlap index...")
                        calculate_overlap_index(image_paths, output_dir, idx + 1)
            else:
                print("No grouped wavefronts found.")
        else:
            print(f"CSV file not found at {os.path.abspath(csv_file_path)}.")
    else:
        print("No directory selected.")

if __name__ == "__main__":
    open_directory()