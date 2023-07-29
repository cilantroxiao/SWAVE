import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Function to parse input
def parse_wave_ids(input_str):
    wave_ids = []
    for part in input_str.split(','):
        if '-' in part:
            start, end = part.split('-')[0], part.split('-')[1]
            wave_ids.extend(range(int(start), int(end) + 1))
        else:
            wave_ids.append(int(part))
    return wave_ids
    
# Argument parser 
parser = argparse.ArgumentParser(
    prog='Polar Direction Plotter',
    description='Plots slow-wave direction data on a polar histogram',
    epilog='Hope this works')
parser.add_argument('filename_wave_ids', nargs='+', metavar='filename:wave_ids', help='input filenames and associated wavefronts with format - \'filename:wave_ids\'')
args = parser.parse_args()

# Master list containing every x and y coord based on parameter/argument
all_directionY = []
all_directionX = []

# Lists for normalized averages after taking individual averages
avg_x_normalized = []
avg_y_normalized = []

data = []

for entry in args.filename_wave_ids:
    filename, wave_ids_input = entry.split(':')
    wave_ids = parse_wave_ids(wave_ids_input)
    data.append({'filename': filename, 'wave_ids': wave_ids})

for entry in data:
    filename = entry['filename']
    wave_ids = entry['wave_ids']
    df = pd.read_csv(Path(f"D:\\ANESTH_{filename}_KX_16\\stage05_channel-wave_characterization\\direction_local\\wavefronts_direction_local.csv"))

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
angles = np.arctan2(all_directionY, all_directionX)
angles_w_avg = np.arctan2(avg_y_normalized, avg_x_normalized)

# Calculate the weighted average angle
weighted_average_angle = np.arctan2(np.average(all_directionY, axis=0), 
                                    np.average(all_directionX, axis=0))
weighted_average_angle1 = np.arctan2(np.average(avg_y_normalized, axis=0), 
                                    np.average(avg_x_normalized, axis=0))
print('weighted with all vectors together', weighted_average_angle)
print('waves normalized individually then normalized together', weighted_average_angle1)

# Create a polar histogram
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.hist(angles, bins=36, range=(-np.pi, np.pi), density=True)

fig1, ax1 = plt.subplots(subplot_kw={'projection': 'polar'})
ax1.hist(angles_w_avg, bins=36, range=(-np.pi, np.pi), density=True)
#print(wave_ids, np.rad2deg(angles_w_avg))

# Plot the weighted average line
ax.plot([0, weighted_average_angle], [0, ax.get_ylim()[1]], color='red', linewidth=2)
ax1.plot([0, weighted_average_angle1], [0, ax1.get_ylim()[1]], color='black', linewidth=2)

ax.set_title('weighted with all vectors together')
ax1.set_title('waves normalized individually then normalized together')

plt.show()
