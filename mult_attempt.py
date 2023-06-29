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
parser.add_argument('--wave_ids', type=int, nargs='+', help='input desired wavefronts (default is all)')
args = parser.parse_args()

all_directionY = []
all_directionX = []

for i, filename in enumerate(args.filenames):
    df = pd.read_csv(Path(filename))

    wave_ids = args.wave_ids if args.wave_ids else df['wavefronts_id'].unique()

    for wave_id in wave_ids:
        group = df[df['wavefronts_id'] == wave_id]

        directionY = group['direction_local_y'].tolist()
        directionX = group['direction_local_x'].tolist()

        all_directionY.extend(directionY)
        all_directionX.extend(directionX)

# Calculate angles using arctan2
angles = np.arctan2(all_directionY, all_directionX)

# Create a polar histogram
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.hist(angles, bins=36, range=(-np.pi, np.pi), density=True)

# Calculate the average angle
average_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
print(f"Average Angle: {average_angle}")

# Plot the average line
ax.plot([0, average_angle], [0, ax.get_ylim()[1]], color='red', linewidth=2)

plt.show()
