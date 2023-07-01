import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(
    prog='Polar Direction Plotter',
    description='Plots slow-wave direction data on a polar histogram',
    epilog='Hope this works')
parser.add_argument('filenames', metavar='CSV', nargs='+', help='input wavefronts_direction_local')
parser.add_argument('--wave_ids', type=int, metavar='N', nargs='*', action='append', help='input desired wavefronts')
args = parser.parse_args()

all_directionY = []
all_directionX = []
weights = []

def weight():
    waves = []
    for j, filename in enumerate(args.filenames):
        df = pd.read_csv(Path(filename))

        last_number = df['wavefronts_id'].iloc[-1]
        waves.append(last_number)

    goal = max(waves)

    #For each wave id != to the max
        # find the commmon factor and multiply it by each file's max length so that each file weighs the same


for i, filename in enumerate(args.filenames):
    df = pd.read_csv(Path(filename))

    last_number = df['wavefronts_id'].iloc[-1]

    wave_ids = args.wave_ids[i] if args.wave_ids else df['wavefronts_id'].unique()
    wave_id = ' '.join(map(str, wave_ids))

    for wave_id in wave_ids:
        group = df[df['wavefronts_id'] == wave_id]

        directionY = group['direction_local_y'].tolist()
        directionX = group['direction_local_x'].tolist()

        all_directionY.extend(directionY)
        all_directionX.extend(directionX)

        weights.extend([len(directionY)] * len(directionX))  # Using length as weights

# Calculate angles using arctan2
angles = np.arctan2(all_directionY, all_directionX)

# Calculate the weighted average angle
weighted_average_angle = np.arctan2(np.average(np.sin(angles), weights=weights),
                                    np.average(np.cos(angles), weights=weights))
print(f"Weighted Average Angle: {weighted_average_angle}")

# Create a polar histogram
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.hist(angles, bins=36, range=(-np.pi, np.pi), density=True)

# Plot the weighted average line
ax.plot([0, weighted_average_angle], [0, ax.get_ylim()[1]], color='red', linewidth=2)

plt.show()
