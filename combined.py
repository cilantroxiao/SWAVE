import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.colors as mcolors
import glob
import csv
import os
import argparse

grid_size = 128
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
    #print(max(wave_ids))
    return wave_ids
  
# Argument parser 
parser = argparse.ArgumentParser(
    prog='Title',
    description='Description',
    epilog='Hope this works')
parser.add_argument('filename_wave_ids', metavar='filename:wave_ids', help='input filenames and associated wavefronts with format - \'filename:wave_ids\'')
parser.add_argument('--norm', action='store_true', help='normalize data?')
parser.add_argument('--avg', action='store_true', help='avg data at end')
parser.add_argument('--out', required=True, help='input output dir')
args = parser.parse_args()

global filename
#Loop through input of 'filename:wave_ids' and check for specific waves or all waves
if ':' in args.filename_wave_ids:    
    filename, wave_ids_input = args.filename_wave_ids.split(':')
    wave_ids = parse_wave_ids(wave_ids_input)
else:
    filename = args.filename_wave_ids
    df = pd.read_csv(Path(f"D:\{filename}\stage05_channel-wave_characterization\direction_local\\wavefronts_direction_local.csv"))
    last_wave_id = df['wavefronts_id'].max()
    wave_ids_input = f'1-{last_wave_id}'  # Set wave_ids_input as a range from 1 to last_wave_id
    wave_ids = parse_wave_ids(wave_ids_input)

#helper functions
def add(list, row, column, df):
    for file in list:
        df_file = pd.read_csv(file)
        if row < df_file.shape[0] and column < df_file.shape[1]:
            cell_value = df_file.iloc[row, column]
            print(cell_value)
            df.iloc[row, column] += cell_value
def divide(df, size):
    for index, row in enumerate(df):
        for col_index, col in enumerate(df[index]):
            df[index][col_index] /= size
    return df
def similar(a , b):
    for state in states:
        if state in str(a):
            a = a.replace(state, '')
        if state in str(b):
            b = b.replace(state, '')
    s = SequenceMatcher(None, a, b)
    mice.append(a)
    if s.ratio() == 1:
        print(f"{a} same mouse")
        return True
    else:
        return False
def polar_histogram(filename):
    # Master list containing every x and y coord based on parameter/argument
    all_directionY = []
    all_directionX = []

    # Lists for normalized averages after taking individual averages
    avg_x_normalized = []
    avg_y_normalized = []
    #Loop and extract data to normalize and calculate average values
    df = pd.read_csv(Path(f"D:\{filename}\stage05_channel-wave_characterization\direction_local\\wavefronts_direction_local.csv"))
    
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

    #fig.savefig(os.path.join(args.out, 'figure1.png'))
    #fig1.savefig(os.path.join(args.out, 'figure2.png'))
    plt.savefig(Path(f"{args.out}\\{os.path.splitext(filename)[0]}_polar.png"))
    plt.close()
def Individual_CSVs(filename):
    filename = filename.strip()
    df = pd.read_csv(Path(f"D:\\{filename}\\stage05_channel-wave_characterization\\channel-wise_measures.csv"))
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
        with open(Path(f"{args.out}\\{filename}_velocity_N.csv"), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(grid)
    #if norm not flagged
    else: 
        for index, row in enumerate(grid):
            for col_index, col in enumerate(row):
                if channel_wave_count[index][col_index] != 0:
                    grid[index][col_index] /= channel_wave_count[index][col_index]
        with open(Path(f"{args.out}\\{filename}_velocity.csv"), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(grid)
    print(f"{filename} done")
def Heat_Mapper(norm=False, avg=False):
    if norm and avg:
        return
    if norm:
        csv_files = glob.glob(f'{args.out}\\*velocity_N.csv')
        scale_min = 0
        scale_max = 2
    elif avg:
        csv_files = glob.glob(f'{args.out}\\*v-avg*.csv')
        scale_min = 0
        scale_max = 2
    else:
        csv_files = glob.glob(f'{args.out}\\*velocity.csv')
        scale_min = 43518.26345849037
        scale_max = 2272431268.2241783
    
    for file in csv_files:
        df = pd.read_csv(file, header=None)
        file_name = os.path.basename(file)
        num_rows = len(df)
        num_columns = len(df.columns)

        x = np.array([i for i in range(num_columns)])
        y = np.array([i for i in range(num_rows)])
        row_shape, col_shape = df.shape
        heat_data = np.zeros((row_shape, col_shape))
        #retrieve data from csv
        for row_index, row in df.iterrows():
            for col_index, col in enumerate(row):
                heat_data[row_index, col_index] = float(row[col_index])

        fig, ax = plt.subplots()

        cmap_nonzero = plt.get_cmap('rainbow')
        cmap_custom = mcolors.ListedColormap(['white'] + [cmap_nonzero(i) for i in range(1, cmap_nonzero.N)])
        norm = mcolors.Normalize(vmin= scale_min, vmax=scale_max)
        heatmap = ax.imshow(heat_data, cmap=cmap_custom, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], norm=norm)
        ax.set_frame_on(False)
        fig.colorbar(heatmap)
        ax.set_aspect('equal')
        ax.set_title(f'{os.path.splitext(file_name)[0]}')
        plt.savefig(Path(f"{args.out}\\{os.path.splitext(file_name)[0]}_heat.png"))
        plt.close()
#need to fix
def avg_planarity():
    table = []
    for i in range(7):
        table.append([])
        for j in range(3):
            table[i].append(0)
    j=0
    for i, file in enumerate(data):
        filename = file.strip()
        df = pd.read_csv(Path(f"D:\\{filename}\\stage05_wave_characterization\\label_planar\\wavefronts_label_planar.csv"), usecols=['planarity'])
        mean = df['planarity'].mean()
        if 'WAKE' in str(filename):
            table[j][0] = mean
        elif 'NREM' in str(filename):
            table[j][1] = mean
        elif 'REM' in str(filename):
            table[j][2] = mean
        if i + 1 < len(data): #if not similar, goes to next list in list
            if not similar(data[i], data[i+1]):
                j += 1

    df = pd.DataFrame({'States': states})
    mice_unique = []
    [mice_unique.append(item) for item in mice if item not in mice_unique]
    for index, column in enumerate(table):
        df.insert(index, mice_unique[index], table[index], True)
    state = df.pop('States')
    df.insert(0, state.name, state)
    df.to_csv(Path("D:\\Sandro_Code\\planarity\\avg_planarity.csv"), index = False, mode='w+')
    print(df)

    #individual bar graphs
    for mouse in mice_unique:
        y = ['WAKE', 'NREM', 'REM']
        x = df[mouse]
        bgraph = plt.bar(y,x)
        bgraph[0].set_color('red')
        bgraph[1].set_color('blue')
        bgraph[2].set_color('green')
        plt.ylabel('average planarity')
        plt.title(mouse)
        plt.savefig(f"D:\\Sandro_Code\\planarity\\{mouse}_avg_planarity.png")
    plt.clf()

    #line plot
    for mouse in mice_unique:
        y = ['WAKE', 'NREM', 'REM']
        x = df[mouse]
        x_masked = np.where(x == 0, np.nan, x)
        plt.plot(y, x_masked, label=mouse, marker='o', linewidth=2)
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')
    plt.title('Average Planarity Across States Comparison')
    plt.tight_layout()
    plt.savefig('D:\\Sandro_Code\\planarity\\avg_planarity_comparison.png')
    plt.clf()
polar_histogram(filename)
Individual_CSVs(filename)
Heat_Mapper()