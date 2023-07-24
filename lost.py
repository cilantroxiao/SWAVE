import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
    
parser = argparse.ArgumentParser(
    prog='Polar Direction Plotter',
    description='Plots slow-wave direction data on a polar histogram',
    epilog='Hope this works')
parser.add_argument('filenames', nargs='+', help='input wavefronts_direction_local')
parser.add_argument('--wave_ids', type=int, metavar='N', nargs='*', action='append', help='input desired wavefronts')
args = parser.parse_args()

#master list containing every x and y coord based on parameter/argument
all_directionY = []
all_directionX = []

#lists for normalized averages after taking individual averages
avg_x_normalized = []
avg_y_normalized = []

for i, filename in enumerate(args.filenames):
    df = pd.read_csv(Path(f"D:\\ANESTH_{filename}_KX_16\\stage05_channel-wave_characterization\\direction_local\\wavefronts_direction_local.csv"))

    wave_ids = args.wave_ids[i] if args.wave_ids else df['wavefronts_id'].unique()
   
    wave_id = ' '.join(map(str, wave_ids))
    last_number = df['wavefronts_id'].iloc[-1]

    for wave_id in wave_ids:
        group = df[df['wavefronts_id'] == wave_id]

        directionY = group['direction_local_y'].tolist()
        directionX = group['direction_local_x'].tolist()

        #normalize everything when it's extracted from the csv file
        directionX_normalized = group.apply(lambda row: row.direction_local_x / np.sqrt(row.direction_local_x**2 + row.direction_local_y**2), axis=1)
        directionY_normalized = group.apply(lambda row: row.direction_local_y / np.sqrt(row.direction_local_x**2 + row.direction_local_y**2), axis=1)

        #calculate their individual average
        avg_y = np.average(directionY_normalized)
        avg_x = np.average(directionX_normalized)
    
        # Calculate the normalized values
        norm_x = avg_x / np.sqrt(avg_x**2 + avg_y**2)
        norm_y = avg_y / np.sqrt(avg_x**2 + avg_y**2)

        #each averaged and normalized angle
        avg_x_normalized.append(norm_x)
        avg_y_normalized.append(norm_y)
        
        #Master List of every single x and y coordinate
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
print('weighted with each wave_id\'s average then averaged together', weighted_average_angle1)

# Create a polar histogram
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.hist(angles, bins=36, range=(-np.pi, np.pi), density=True)

fig1, ax1 = plt.subplots(subplot_kw={'projection': 'polar'})
ax1.hist(angles_w_avg, bins = 36, range=(-np.pi, np.pi), density=True)
print(wave_ids, np.rad2deg(angles_w_avg))

# Plot the weighted average line
ax.plot([0, weighted_average_angle], [0, ax.get_ylim()[1]], color='red', linewidth=2)
ax1.plot([0, weighted_average_angle1], [0, ax1.get_ylim()[1]], color='black', linewidth=2)

ax.set_title('weighted with all vectors together')
ax1.set_title('weighed with each wave_id\'s average then averaged together')

plt.show()

#Mouse Names
'''
119_2
132_1
132_2
328A_3
E2
L1
L3
'''