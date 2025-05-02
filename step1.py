import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import csv
import os
import seaborn as sns
import cv2
import tkinter as tk
import datetime
grid_size = 128

#helper functions
def get_recording_length(path_head):
    #retrieve frequency divisor
    frequency_path = f'{path_head}\\stage05_wave_characterization\\annotations\\wavefronts_annotations.csv'
    col = 'recording_length'
    with open(frequency_path, mode='r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        column_index = header.index(col)
        first_row = next(csv_reader)
        frequency = first_row[column_index]
        return float(frequency)

def create_composite_folder(currdir):
    composite_folder = os.path.join(currdir, "composites")
    if not os.path.exists(composite_folder):
        os.makedirs(composite_folder)
        print(f"Created composite images folder: {composite_folder}")
    return composite_folder

def output_1(image):
    edges = cv2.Canny(image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
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
        return image[y:y+h, x:x+w]
    else:
        return image

def calculate_overlap_index(image_paths):
    images = [cv2.imread(img_path, cv2.IMREAD_COLOR) for img_path in image_paths if os.path.exists(img_path)]
    if not images:
        return None, []
    
    min_height = min(img.shape[0] for img in images)
    min_width = min(img.shape[1] for img in images)
    resized_images = [cv2.resize(img, (min_width, min_height)) for img in images]
    
    graph_cropped_images = [output_1(img) for img in resized_images]
    min_height_cropped = min(img.shape[0] for img in graph_cropped_images)
    min_width_cropped = min(img.shape[1] for img in graph_cropped_images)
    aligned_graph_images = [cv2.resize(img, (min_width_cropped, min_height_cropped)) for img in graph_cropped_images]
    
    graph_composite_image = np.sum([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in aligned_graph_images], axis=0).astype(np.float32)
    graph_composite_normalized = cv2.normalize(graph_composite_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return graph_composite_normalized, resized_images

def save_group_composite(output_path, group_index, composite_image, individual_images):
    composite_image_filename = os.path.join(output_path, f'group_{group_index}_composite.png')
    plt.imshow(composite_image, cmap='gray')
    plt.axis('off')
    plt.title(f'Group {group_index} Composite')
    plt.savefig(composite_image_filename)
    plt.close()

def process_wavefronts(csv_file_path, label_planar_path, composite_folder):
    print(f"Processing wavefronts from {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    wave_groups = []
    current_group = [(df.loc[0, 'wavefronts_id'], df.loc[0, 'time_stamp'])]
    
    for i in range(1, len(df)):
        current_id = df.loc[i, 'wavefronts_id']
        current_time = df.loc[i, 'time_stamp']
        previous_time = df.loc[i - 1, 'time_stamp']
        
        if abs(current_time - previous_time) <= 0.5:
            current_group.append((current_id, current_time))
        else:
            wave_groups.append(current_group)
            current_group = [(current_id, current_time)]
    
    wave_groups.append(current_group)
    
    for idx, group in enumerate(wave_groups):
        if len(group) > 1:
            image_paths = [os.path.join(label_planar_path, f"wave_{wave_id}.png") for wave_id, _ in group]
            composite_image, individual_images = calculate_overlap_index(image_paths)
            save_group_composite(composite_folder, idx + 1, composite_image, individual_images)


#visualization functions
def Polar_Histogram(path_head, filename, wave_ids, currdir):
    print(f"Entered Polar_Histogram for {filename}")
    # Master list containing every x and y coord based on parameter/argument
    all_directionY = []
    all_directionX = []
    # Lists for normalized averages after taking individual averages
    avg_x_normalized = []
    avg_y_normalized = []
    # List to add each individual wave's normalized values
    all_directionX_normalized = []
    all_directionY_normalized = [] 
    
    #Extract and Loop data to normalize and calculate average values
    df = pd.read_csv(Path(f"{path_head}\\stage05_channel-wave_characterization\\direction_local\\wavefronts_direction_local.csv"))
    # Master List of every single x and y coordinate
    all_directionY.append(df['direction_local_y'].tolist())
    all_directionX.append(df['direction_local_x'].tolist())
    for wave_id in wave_ids:
        group = df[df['wavefronts_id'] == wave_id]

        # Normalize everything when it's extracted from the csv file
        directionX_normalized = group.apply(lambda row: row.direction_local_x / np.sqrt(row.direction_local_x**2 + row.direction_local_y**2), axis=1)
        directionY_normalized = group.apply(lambda row: row.direction_local_y / np.sqrt(row.direction_local_x**2 + row.direction_local_y**2), axis=1)
        # Calculate their individual average
        avg_y = np.average(directionY_normalized)
        avg_x = np.average(directionX_normalized)
        # Calculate the normalized values
        norm_x = avg_x / np.sqrt(avg_x**2 + avg_y**2)
        norm_y = avg_y / np.sqrt(avg_x**2 + avg_y**2)

        # Master List of every single normalized x and y coordinate
        all_directionX_normalized.extend(directionX_normalized)
        all_directionY_normalized.extend(directionY_normalized)
        # Master List of each averaged and normalized angle
        avg_x_normalized.append(norm_x)
        avg_y_normalized.append(norm_y)       

    # Calculate angles using arctan2
    angles_norm = np.arctan2(all_directionY_normalized, all_directionX_normalized)
    angles_avg_norm = np.arctan2(avg_y_normalized, avg_x_normalized)  

    # Calculate the weighted average angle
    weighted_average_angle_norm = np.arctan2(np.average(all_directionY_normalized), 
                                        np.average(all_directionX_normalized))
    weighted_average_angle_avg_norm = np.arctan2(np.average(avg_y_normalized), 
                                        np.average(avg_x_normalized))

    # Create a polar histogram each method
    fig, (ax1, ax2) = plt.subplots(1,2, subplot_kw={'projection': 'polar'})

    ax1.hist(angles_norm, bins=36, range=(-np.pi, np.pi), density=True)
    ax1.set_title('All waves have equal weight')

    ax2.hist(angles_avg_norm, bins=36, range=(-np.pi, np.pi), density=True)
    ax2.set_title('Each wave weighted by # of vectors')

    # Plot the weighted average line
    ax1.plot([0, weighted_average_angle_norm], [0, ax1.get_ylim()[1]], color='red', linewidth=2)
    ax2.plot([0, weighted_average_angle_avg_norm], [0, ax2.get_ylim()[1]], color='black', linewidth=2)

    fig.tight_layout()
    fig_path = os.path.join(currdir, f'{filename}_polar.png')
    fig.savefig(fig_path)
    plt.close()

def Velocity_CSVs(path_head, filename, wave_ids, currdir):
    print(f"Entered Velocity_CSVs for {filename}")

    df = pd.read_csv(Path(f"{path_head}\\stage05_channel-wave_characterization\\channel-wise_measures.csv"))
    
    #create empty 128x128 grid for velocities
    grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]

    #fill grid with velocities
    for i, row in df.iterrows():
        if row['wavefronts_id'] in wave_ids:
            x = int(row['x_coords'])
            y = int(row['y_coords'])
            velocity = float(row['velocity_local'])      
            grid[y][x] += velocity

    #write csv file
    with open(os.path.join(currdir, f'{filename}_velocity.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(grid)

def Planarity(path_head, filename, wave_ids, currdir):
    print(f"Entered Planarity for {filename}")

    df = pd.read_csv(Path(f"{path_head}\\stage05_wave_characterization\\label_planar\\wavefronts_label_planar.csv"))

    #calculate planarity mean
    filtered_df = df[df['wavefronts_id'].isin(wave_ids)]
    mean = filtered_df['planarity'].mean()

    #write planarity file
    with open(os.path.join(currdir, f'{filename}_planarity.csv'), 'w') as f:
        f.write(str(mean))

def Num_Waves(path_head, filename, wave_ids, currdir):
    print(f"Entered Num_Waves for {filename}")

    df = pd.read_csv(Path(f"{path_head}\\stage05_channel-wave_characterization\\velocity_local\\wavefronts_velocity_local.csv"), usecols=['wavefronts_id'])
    
    #retrieve number of waves by grabbing tail value of wavefronts_id column
    index = df.tail(1).index.item() 
    num = df.at[index, 'wavefronts_id']

    #verify num value
    if num != len(wave_ids):
        num = len(wave_ids)

    #write numwaves file
    numwaves_file = f'{filename}_numwaves.csv'
    with open(os.path.join(currdir, numwaves_file), 'w') as f:
        f.write(str(num))

def Num_Waves_Topo(path_head, filename, currdir):
    print(f"Entered Num_Waves_Topo for {filename}")

    #initialize
    channel_list = np.zeros(grid_size**2)
    dir = f'{path_head}\\stage05_channel-wave_characterization\\direction_local\\wavefronts_direction_local.csv'
    df = pd.read_csv(dir)

    #count number of waves per channel
    for channel in df['channel_id']:
        channel_list[channel] += 1

    #reshape to 2d array and rotate
    grid = channel_list.reshape(-1, grid_size)
    rotated_grid = np.rot90(grid)

    #csv
    np.savetxt(os.path.join(currdir, f'{filename}_numwaves_topo.csv'), rotated_grid, delimiter=',')
    
    #topo paramters
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(rotated_grid, 
                            annot=False, 
                            cmap= "viridis", 
                            robust=True, 
                            xticklabels=False, 
                            yticklabels=False, 
                            square=True, 
                            cbar=True)
    heatmap.tick_params(left=False, bottom=False)
    heatmap.set_aspect('equal')
    heatmap.invert_yaxis()
    heatmap.set_title(f'{filename} number of waves', fontsize=18, loc="center")

    #save topo
    plt.savefig(os.path.join(currdir, f'{filename}_numwaves_topo.png'))
    plt.close()

def Freq_Waves(path_head, filename, currdir):  
    print(f"Entered Freq_Waves for {filename}")

    #retrieve frequency divisor
    frequency = get_recording_length(path_head)

    #divide numwaves file by frequency
    source_file_path = os.path.join(currdir, f'{filename}_numwaves.csv')
    num_data = np.genfromtxt(source_file_path, delimiter=',', dtype=float)
    freq_data = num_data / frequency

    #write and save numwaves file
    freqwaves_file = f'{filename}_freqwaves.csv'
    freqwaves_path = os.path.join(currdir,freqwaves_file)
    with open(freqwaves_path, 'w') as f:
       f.write(str(freq_data))

def Freq_Waves_Topo(path_head, filename, currdir):
    print(f"Entered Freq_Waves_Topo for {filename}")

    #retrieve frequency divisor
    frequency = get_recording_length(path_head)

    #divide numwaves file by frequency
    source_file_path = os.path.join(currdir, f'{filename}_numwaves_topo.csv')
    num_data = np.genfromtxt(source_file_path, delimiter=',', dtype=float)
    freq_data = num_data / frequency

    #csv
    np.savetxt(os.path.join(currdir, f'{filename}_freqwaves_topo.csv'), freq_data, delimiter=',')

    #topo parameters
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(freq_data, 
                            annot=False, 
                            cmap= "viridis", 
                            robust=True, 
                            xticklabels=False,
                            yticklabels=False, 
                            square=True, 
                            cbar=True)
    heatmap.tick_params(left=False, bottom=False)
    heatmap.set_aspect('equal')
    heatmap.invert_yaxis()
    heatmap.set_title(f'{filename} wave frequency', fontsize=18, loc="center")

    #save topo
    plt.savefig(os.path.join(currdir, f'{filename}_freqwaves_topo.png'))
    plt.close()

def run(data_path, filename, wave_ids, args, currdir):
    path_head = os.path.join(data_path, filename)
    stage05_path = os.path.join(path_head, 'stage05_wave_characterization', 'time_stamp')
    csv_file_path = os.path.join(stage05_path, 'wavefronts_time_stamp.csv')
    label_planar_path = os.path.join(path_head, 'stage05_wave_characterization', 'label_planar')
    
    
    composite_folder = create_composite_folder(currdir)
    process_wavefronts(csv_file_path, label_planar_path, composite_folder)
    # Polar_Histogram(path_head, filename, wave_ids, currdir)
    # Velocity_CSVs(path_head, filename, wave_ids, currdir)
    # Planarity(path_head, filename, wave_ids, currdir)
    # Num_Waves(path_head, filename, wave_ids, currdir)
    # Num_Waves_Topo(path_head, filename, currdir)
    
    # #freq flag
    # if args.freq:
    #     Freq_Waves(path_head, filename,currdir)
    #     Freq_Waves_Topo(path_head, filename, currdir)