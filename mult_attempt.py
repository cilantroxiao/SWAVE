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
parser.add_argument('--wave_ids', type=int, metavar='N', nargs='*', help='input desired wavefronts (default is all)')
args = parser.parse_args()

wave_ids_per_file = {}
for filename in args.filenames:
    wave_ids_per_file[filename] = args.wave_ids

for filename, wave_ids in wave_ids_per_file.items():
    df = pd.read_csv(Path(filename))

    if len(wave_ids) == 0:
        last_wave = df['wavefronts_id'].values[-1]
        wave_ids = [i for i in range(last_wave + 1)]
        wave_id = f"All {last_wave}"
    else:
        wave_id = ' '.join(map(str, wave_ids))

    directionY, directionX = [], []
    for index, row in df.iterrows():
        if row['wavefronts_id'] in wave_ids:
            directionY.append(row['direction_local_y'])
            directionX.append(row['direction_local_x'])

    # Calculate angles using arctan2
    angles = np.arctan2(directionY, directionX)

    # Create a polar histogram
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.hist(angles, bins=36, range=(-np.pi, np.pi), density=True)

    # Calculate the average angle
    average_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    print(f"Wavefronts ID: {wave_id}, Average Angle: {average_angle}")

    # Plot the average line
    ax.plot([0, average_angle], [0, ax.get_ylim()[1]], color='red', linewidth=2)

    plt.show()
