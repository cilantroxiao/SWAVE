import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Function to parse input
def parse_wave_ids(wave_ids_input):
    wave_ids = []
    if wave_ids_input:
        for part in wave_ids_input.split(','):
            if '-' in part:
                start, end = part.split('-')[0], part.split('-')[1]
                wave_ids.extend(range(int(start), int(end) + 1))
            else:
                wave_ids.append(int(part))
    return wave_ids

# Read input filenames and wave_ids from the text file
input_file = 'C:\\Users\\Eric\\Documents\\input_files.txt'
with open(input_file, 'r') as file:
    lines = file.readlines()

data = []

# Master list containing every x and y coord based on parameter/argument
all_directionY = []
all_directionX = []

# Lists for normalized averages after taking individual averages
avg_x_normalized = []
avg_y_normalized = []

# Loop through input file lines and extract data
for line in lines:
    entry = line.strip()
    if ':' in entry:
        filename, wave_ids_input = entry.split(':')
        wave_ids = parse_wave_ids(wave_ids_input)
        data.append({'filename': filename, 'wave_ids': wave_ids})
    else:
        filename = entry
        df = pd.read_csv(Path(f"D:\{filename}\stage05_channel-wave_characterization\direction_local\\wavefronts_direction_local.csv"))
        last_wave_id = df['wavefronts_id'].max()
        wave_ids_input = f'1-{last_wave_id}'  # Set wave_ids_input as a range from 1 to last_wave_id
        wave_ids = parse_wave_ids(wave_ids_input)
        data.append({'filename': filename, 'wave_ids': wave_ids})

#Loop and extract data to normalize and calculate average values
for entry in data:
    filename = entry['filename']
    wave_ids = entry['wave_ids']
    df = pd.read_csv(Path(f"D:\\{filename}\\stage05_channel-wave_characterization\\direction_local\\wavefronts_direction_local.csv"))
    
    for wave_id in wave_ids:
        group = df[df['wavefronts_id'] == wave_id]

        directionY = group['direction_local_y'].tolist()
        directionX = group['direction_local_x'].tolist()

        # Normalize everything when it's extracted from the csv file
        directionX_normalized = group.apply(lambda row: row.direction_local_x / np.sqrt(row.direction_local_x**2 + row.direction_local_y**2), axis=1)
        directionY_normalized = group.apply(lambda row: row.direction_local_y / np.sqrt(row.direction_local_x**2 + row.direction_local_y**2), axis=1)

        # Calculate their individual average
        avg_y = np.average(directionY_normalized)
        avg_x = np.average(directionX_normalized)
    
        # Calculate the normalized values
        norm_x = avg_x / np.sqrt(avg_x**2 + avg_y**2)
        norm_y = avg_y / np.sqrt(avg_x**2 + avg_y**2)

        # Each averaged and normalized angle
        avg_x_normalized.append(norm_x)
        avg_y_normalized.append(norm_y)
        
        # Master List of every single x and y coordinate
        all_directionY.extend(directionY)
        all_directionX.extend(directionX)

# Calculate angles using arctan2
angles = np.arctan2(directionY_normalized, directionX_normalized)
angles_norm = np.arctan2(avg_y_normalized, avg_x_normalized)

# Calculate the weighted average angle
weighted_average_angle = np.arctan2(np.average(directionY_normalized), 
                                    np.average(directionX_normalized))
weighted_average_angle1 = np.arctan2(np.average(avg_y_normalized), 
                                    np.average(avg_x_normalized))
print('normalized by vector length', weighted_average_angle)
print('normalized by number of vectors within each wave', weighted_average_angle1)

# Create a polar histogram each method
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.hist(angles, bins=36, range=(-np.pi, np.pi), density=True)
ax.set_title('normalized by vector length')

fig1, ax1 = plt.subplots(subplot_kw={'projection': 'polar'})
ax1.hist(angles_norm, bins=36, range=(-np.pi, np.pi), density=True)
ax1.set_title('normalized by number of vectors within each wave')

# Plot the weighted average line
ax.plot([0, weighted_average_angle], [0, ax.get_ylim()[1]], color='red', linewidth=2)
ax1.plot([0, weighted_average_angle1], [0, ax1.get_ylim()[1]], color='black', linewidth=2)

plt.show()