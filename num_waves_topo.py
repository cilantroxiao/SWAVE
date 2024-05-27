import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.colors as mcolors
from difflib import SequenceMatcher
import glob
import csv
import os
import warnings
import seaborn as sns

grid_size = 128
channel_list = np.zeros(grid_size**2)
dir = 'D:\\SLEEP_132_1_REM_54\\stage05_channel-wave_characterization\\direction_local\\wavefronts_direction_local.csv'

df = pd.read_csv(dir)

for channel in df['channel_id']:
    channel_list[channel] += 1
grid = channel_list.reshape(-1, grid_size)
rotated_grid = np.rot90(grid)

#csv
file_name = "num_waves_topo.csv"
np.savetxt(f".\\output\\{file_name}.csv", rotated_grid, delimiter=',')

#heatmap
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(rotated_grid, 
                        annot=False, 
                        cmap= "viridis", robust=True, xticklabels=False, yticklabels=False, square=True, cbar=True)
heatmap.tick_params(left=False, bottom=False)
heatmap.set_aspect('equal')
heatmap.invert_yaxis()
heatmap.set_title(file_name, fontsize=18, loc="center")
plt.savefig(f".\\output\\{file_name}.png")

    
