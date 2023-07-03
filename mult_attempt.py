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
all_weights = []     

def weightCalc(goal, lastnum, df):
    weights = []
    factor = lastnum // goal  # Find the common factor
    weight = factor * goal  # Multiply by max length
    for i in range(len(df)):
        weights.append(weight)
    return weights

amountOfWaves = []
for j, filename in enumerate(args.filenames):
    df = pd.read_csv(Path(filename))
    last = df['wavefronts_id'].iloc[-1]
    amountOfWaves.append(last)
goals = max(amountOfWaves)


for i, filename in enumerate(args.filenames):
    df = pd.read_csv(Path(filename))

    last_number = df['wavefronts_id'].iloc[-1]

   

    wave_ids = args.wave_ids[i] if args.wave_ids else df['wavefronts_id'].unique()
    wave_id = ' '.join(map(str, wave_ids))

    for wave_id in wave_ids:
        group = df[df['wavefronts_id'] == wave_id]

        directionY = group['direction_local_y'].tolist()
        directionX = group['direction_local_x'].tolist()

        weights = weightCalc(goals, last_number, df)

        all_directionY.extend(directionY)
        all_directionX.extend(directionX)

        all_weights.extend(weights) 

# Calculate angles using arctan2
angles = np.arctan2(all_directionY, all_directionX)
print(len(angles))
print(len(all_weights))
#PROBLEM NOW IS THAT THE LENGTHS OF THE TWO LISTS AREN'T THE SAME


# Calculate the weighted average angle
weighted_average_angle = np.arctan2(np.average(np.sin(angles), weights=all_weights, axis=0), 
                                    np.average(np.cos(angles), weights=all_weights, axis=0))  
print(f"Weighted Average Angle: {weighted_average_angle}")


# Create a polar histogram
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.hist(angles, bins=36, range=(-np.pi, np.pi), density=True)

# Plot the weighted average line
ax.plot([0, weighted_average_angle], [0, ax.get_ylim()[1]], color='red', linewidth=2)

plt.show()
