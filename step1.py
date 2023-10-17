import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.colors as mcolors
from difflib import SequenceMatcher
import glob
import csv
import os
import argparse
grid_size = 128
states = ['WAKE', 'NREM', 'REM']
mice = []
# Argument parser 
def create_parser():
    parser = argparse.ArgumentParser(
        prog='Title',
        description='Description',
        epilog='Hope this works')
    parser.add_argument('filename_wave_ids', help='input filenames and associated wavefronts')
    parser.add_argument('--out', help='input output dir', required = True)
    parser.add_argument('--norm', action='store_true', help='normalized csv?')
    return parser
#Loop through input of 'filename:wave_ids' and check for specific waves or all waves
def parse_waves():
    wave_ids = list()
    if ':' in args.filename_wave_ids: #if waves/ranges specified
        filename, wave_ids_input = args.filename_wave_ids.split(':')
        for part in wave_ids_input.split(','):
            if '-' in part:
                start, end = part.split('-')[0], part.split('-')[1]
                wave_ids.extend(range(int(start), int(end) + 1))
            else:
                wave_ids.append(int(part))
    else: #all waves
        filename = args.filename_wave_ids
        df = pd.read_csv(Path(f"D:\{filename}\stage05_channel-wave_characterization\direction_local\\wavefronts_direction_local.csv"))
        last_wave_id = df['wavefronts_id'].max()
        wave_ids.extend(range(1, last_wave_id)) # Set wave_ids_input as a range from 1 to last_wave_id
                
    return filename, wave_ids
#helper functions
def add(list, row, column, df):
    for file in list:
        df_file = pd.read_csv(file)
        if row < df_file.shape[0] and column < df_file.shape[1]:
            cell_value = df_file.iloc[row, column]
            
            df.iloc[row, column] += cell_value
def divide(df, size):
    for index, row in enumerate(df):
        for col_index, col in enumerate(df[index]):
            if size != 0 and not np.isnan(col):
                df.at[index,col_index] /= size
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
        return True
    else:
        return False
def Polar_Histogram(filename, wave_ids):
    # Master list containing every x and y coord based on parameter/argument
    all_directionY = []
    all_directionX = []

    # Lists for normalized averages after taking individual averages
    avg_x_normalized = []
    avg_y_normalized = []
    #Loop and extract data to normalize and calculate average values
    df = pd.read_csv(Path(f"D:\{filename}\stage05_channel-wave_characterization\direction_local\\wavefronts_direction_local.csv"))
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
    print(f"{args.out}\\{filename}_polar.png")
    plt.savefig(Path(f"{args.out}\\{filename}_polar.png"))
    
    plt.close()
def Individual_CSVs(filename, wave_ids):
    filename = filename.strip()
    df = pd.read_csv(Path(f"D:\\{filename}\\stage05_channel-wave_characterization\\channel-wise_measures.csv"))
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
        print(f"Producing normalized heatmaps...")
        csv_files = glob.glob(f'{args.out}\\*velocity_N.csv')
        scale_min = 0
        scale_max = 2
    elif avg:
        print(f"Producing average heatmaps...")
        csv_files = glob.glob(f'{args.out}\\*v-avg*.csv')
        scale_min = 0
        scale_max = 2
    else:
        print(f"Producing velocity heatmaps...")
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
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    filename, wave_ids = parse_waves()
    Polar_Histogram(filename, wave_ids)
    Individual_CSVs(filename, wave_ids)
    Heat_Mapper(False, False)