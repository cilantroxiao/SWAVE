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
grid_size = 128
states = ['WAKE', 'NREM', 'REM']
mice = []

#helper functions
def add(list, row, column, df):
    for file in list:
        df_file = pd.read_csv(file)
        if row < df_file.shape[0] and column < df_file.shape[1]:
            cell_value = int(df_file.iloc[row, column])
            
            df.iloc[row, column] += cell_value
def divide(df, size):
    for index, row in enumerate(df):
        for col_index, col in enumerate(df[index]):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                if size != 0 and not np.isnan(col):
                    df.at[index,col_index] /= size
    return df
def similar(curr, next):
    for state in states:
        if state in str(curr):
            curr = curr.replace(state, '')
        if state in str(next):
            next = next.replace(state, '')
    s = SequenceMatcher(None, curr, next)
    #if curr not in mice:
    #    mice.append(curr)
    if s.ratio() == 1:
        return True
    else:
        return False
def check_mice(csv_files, flagType):
    for file in csv_files[:]:  # Using a copy of the list to avoid modifying it during iteration
        print(file)
        file_name = os.path.splitext(os.path.basename(file))[0].replace(flagType, '')
        file_name = file_name.replace('_velocity', '')
        print(file_name)
        found = False
        with open('.\\input.txt') as f:
            for line in f:
                line = line.split(' ')
                for item in line:
                    if '_' in item:
                        if ':' in item:
                            name, wave = item.split(':')
                            if file_name == name:
                                found = True
                                break
                            else:
                                found = False
                        else:
                            name = item
                            if file_name == name:
                                found = True
                                break
                            else:
                                found = False
        if not found:
            print("This mouse is not in the input.txt file:", file_name)
            csv_files.remove(file)

def Polar_Histogram(path_head, filename, wave_ids, currdir):
    # Master list containing every x and y coord based on parameter/argument
    all_directionY = []
    all_directionX = []

    # Lists for normalized averages after taking individual averages
    avg_x_normalized = []
    avg_y_normalized = []
    #Loop and extract data to normalize and calculate average values
    df = pd.read_csv(Path(f"{path_head}\\stage05_channel-wave_characterization\\direction_local\\wavefronts_direction_local.csv"))
    print(f"Graphing {filename} polar histogram...")
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
    #print('normalized by vector length', weighted_average_angle)
    #print('normalized by number of vectors within each wave', weighted_average_angle1)

    # Create a polar histogram each method
    fig, (ax, ax1) = plt.subplots(1,2, subplot_kw={'projection': 'polar'})
    ax.hist(angles, bins=36, range=(-np.pi, np.pi), density=True)
    ax.set_title('normalized by vector length')

    ax1.hist(angles_norm, bins=36, range=(-np.pi, np.pi), density=True)
    ax1.set_title('normalized by # vectors in wave')
    
    # Plot the weighted average line
    ax.plot([0, weighted_average_angle], [0, ax.get_ylim()[1]], color='red', linewidth=2)
    ax1.plot([0, weighted_average_angle1], [0, ax1.get_ylim()[1]], color='black', linewidth=2)

    #print(f'{currdir}\\{filename}_polar.png')
    fig.savefig(os.path.join(currdir, f'{filename}_polar.png'))
    plt.close()

def Velocity_CSVs(path_head, filename, wave_ids, args, currdir):
    filename = filename.strip()
    df = pd.read_csv(Path(f"{path_head}\\stage05_channel-wave_characterization\\channel-wise_measures.csv"))
    print(f"Producing {filename} velocities CSV...")
    grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
    channel_wave_count = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
    for index, row in df.iterrows():
        if row['wavefronts_id'] in wave_ids:
            x = int(row['x_coords'])
            y = int(row['y_coords'])
            velocity = float(row['velocity_local'])      
            grid[y][x] += velocity
            channel_wave_count[y][x] += 1
    #if norm flagged
    if args.norm: 
        with open(Path(f"{currdir}\\{filename}_velocity_N.csv"), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(grid)
    #if norm not flagged
    else: 
        for index, row in enumerate(grid):
            for col_index, col in enumerate(row):
                if channel_wave_count[index][col_index] != 0:
                    grid[index][col_index] /= channel_wave_count[index][col_index]
        with open(Path(f"{currdir}\\{filename}_velocity.csv"), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(grid)
    print(f"{filename} done")

def Planarity(path_head, filename, wave_ids, currdir):
    print("Calculating planarity...")
    df = pd.read_csv(Path(f"{path_head}\\stage05_wave_characterization\\label_planar\\wavefronts_label_planar.csv"))
    filtered_df = df[df['wavefronts_id'].isin(wave_ids)]
    mean = filtered_df['planarity'].mean()
    with open(Path(f"{currdir}\\{filename}_planarity.csv"), 'w') as f:
        f.write(str(mean))

def Num_Waves(path_head, filename, wave_ids, currdir):
    print("Retrieving number of waves...")
    df = pd.read_csv(Path(f"{path_head}\\stage05_channel-wave_characterization\\velocity_local\\wavefronts_velocity_local.csv"), usecols=['wavefronts_id'])
    index = df.tail(1).index.item() #grabs last index's value in file
    num = df.at[index, 'wavefronts_id']
    if num != len(wave_ids):
        num = len(wave_ids)
    with open(Path(f"{currdir}\\{filename}_numwaves.csv"), 'w') as f:
        f.write(str(num))

def run(data_path, filename, wave_ids, args, currdir):
    path_head = os.path.join(data_path, filename)
    Polar_Histogram(path_head, filename, wave_ids, currdir)
    Velocity_CSVs(path_head, filename, wave_ids, args, currdir)
    Planarity(path_head, filename, wave_ids, currdir)
    Num_Waves(path_head, filename, wave_ids, currdir)