import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.colors as mcolors
from difflib import SequenceMatcher
import glob
import csv
from num_waves import similar, data, states, mice_unique
import os
import argparse

parser = argparse.ArgumentParser(
                    prog='Velocity Heat Maps',
                    description='Plots slow-wave average velocity data per channel on a heat map',
                    epilog='Hope this works')
parser.add_argument("--norm", default=False, action="store_true",
                    help="Flag if you want to to normalize values")
#parser.add_argument('wave_ids', type=int, metavar='N', nargs='*', 
#                    help='input desired wavefronts (default is all). Delimit ranges by commas: w-x,y,z-a')
args = parser.parse_args()
test = False
grid_size = 128
out_path = 'D:\\Sandro_Code\\channel_wise_velocity'
#wave_ids = args.wave_ids.split(',')
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
def Individual_CSVs():
    for file in data:
        filename = file.strip()
        df = pd.read_csv(Path(f"D:\\{filename}\\stage05_channel-wave_characterization\\channel-wise_measures.csv"))
        grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
        channel_wave_count = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
        for index, row in df.iterrows():
            x = int(row['x_coords'])
            y = int(row['y_coords'])
            velocity = float(row['velocity_local'])      
            grid[y][x] += velocity
            channel_wave_count[y][x] += 1
        #if norm flagged
        #args.norm
        if test: 
            with open(Path(f"{out_path}\\{filename}_velocity_N.csv"), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(grid)
        #if norm not flagged
        else: 
            for index, row in enumerate(grid):
                for col_index, col in enumerate(row):
                    if channel_wave_count[index][col_index] != 0:
                        grid[index][col_index] /= channel_wave_count[index][col_index]
            with open(Path(f"{out_path}\\{filename}_velocity.csv"), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(grid)
        print(f"{filename} done")
def Average_CSVs():
    grid_average = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
    averages_nrem, averages_rem, averages_wake = pd.DataFrame(grid_average), pd.DataFrame(grid_average), pd.DataFrame(grid_average)
    norm_path = f'{out_path}\\*velocity_N.csv'
    avg_path = f'{out_path}\\*velocity.csv'
    #args.norm
    if test:
        nrem_csv_files = [x for x in glob.glob(norm_path) if 'NREM' in x]
        rem_csv_files = [x for x in glob.glob(norm_path) if 'REM' in x]
        wake_csv_files = [x for x in glob.glob(norm_path) if 'WAKE' in x]
    #if not norm
    else:
        nrem_csv_files = [x for x in glob.glob(avg_path) if 'NREM' in x]
        rem_csv_files = [x for x in glob.glob(avg_path) if 'REM' in x]
        wake_csv_files = [x for x in glob.glob(avg_path) if 'WAKE' in x]

    for row in range(grid_size):
        for column in range(grid_size):
            add(nrem_csv_files, row, column, averages_nrem)
            add(rem_csv_files, row, column, averages_rem)
            add(wake_csv_files, row, column, averages_wake)
    nrem_avg_path = norm_path.replace('*velocity', 'NREM_v-avg')
    rem_avg_path = norm_path.replace('*velocity', 'REM_v-avg')
    wake_avg_path = norm_path.replace('*velocity', 'WAKE_v-avg')
    divide(averages_nrem, len(nrem_csv_files)).to_csv(Path(nrem_avg_path), index = False, header = False, mode='w+')
    divide(averages_rem, len(rem_csv_files)).to_csv(Path(rem_avg_path), index = False, header = False, mode='w+')
    divide(averages_wake, len(wake_csv_files)).to_csv(Path(wake_avg_path), index = False, header = False, mode='w+')
def Heat_Mapper(norm=False, avg=False):
    if norm and avg:
        return
    if norm:
        csv_files = glob.glob(f'{out_path}\\*velocity_N.csv')
        scale_min = 0
        scale_max = 2
    elif avg:
        csv_files = glob.glob(f'{out_path}\\*v-avg*.csv')
        scale_min = 0
        scale_max = 2
    else:
        csv_files = glob.glob(f'{out_path}\\*velocity.csv')
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
        plt.savefig(Path(f"{out_path}\\{os.path.splitext(file_name)[0]}.png"))
        plt.close()
def Normalize():
    unique = (
    'SLEEP_L1__54', 
    'SLEEP_L3__54', 
    'SLEEP_E2__54', 
    'SLEEP_119_2__54', 
    'SLEEP_328A_3__54', 
    'SLEEP_132_1__54', 
    'SLEEP_132_2__54', 
    )
    
    csv_files = glob.glob(f'{out_path}\\*velocity_N.csv')
    j = 0
    sums = [0] * len(unique)
    for i, file in enumerate(csv_files):
        df = pd.read_csv(file, header=None)
        file_name = os.path.splitext(os.path.basename(file))[0]
        sums[j] += df.sum().sum(axis=0)
        if i + 1 < len(csv_files) and not similar(file_name, os.path.splitext(os.path.basename(csv_files[i+1]))[0]):
            j += 1
    denominator = [x / (128**2 * 3) for x in sums]
    j = 0
    for i, file in enumerate(csv_files):
        df = pd.read_csv(file, header=None)
        file_name = os.path.splitext(os.path.basename(file))[0]
        df = df.div(denominator[j])
        df.to_csv(Path(file), index = False, header = False)
        if i + 1 < len(csv_files) and not similar(file_name, os.path.splitext(os.path.basename(csv_files[i+1]))[0]):
            j += 1
Individual_CSVs()
#Average_CSVs()
#if test: 
#    Normalize()
Heat_Mapper()

#done, can delete
def test_non_norm_on_one():
        test = [
        'SLEEP_L1_REM_54',
        'SLEEP_L1_NREM_54',
        'SLEEP_L1_WAKE_54'
        ]
        for file in test:
            filename = file.strip()
            df = pd.read_csv(Path(f"D:\\{filename}\\stage05_channel-wave_characterization\\channel-wise_measures.csv"))
            grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
            channel_wave_count = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
            for index, row in df.iterrows():
                x = int(row['x_coords'])
                y = int(row['y_coords'])
                velocity = float(row['velocity_local'])      
                grid[y][x] += velocity
                channel_wave_count[y][x] += 1
    #commented for norm, keep for not norm
            for index, row in enumerate(grid):
                for col_index, col in enumerate(row):
                    if channel_wave_count[index][col_index] != 0:
                        grid[index][col_index] /= channel_wave_count[index][col_index]
            #with open(Path(f"D:\\Sandro_Code\\channel_wise_velocity\\{filename}_velocity_norm.csv"), 'w', newline='') as csvfile:
            with open(Path(f"D:\\Sandro_Code\\channel_wise_velocity\\{filename}_velocity.csv"), 'w', newline='') as csvfile:   
                writer = csv.writer(csvfile)
                writer.writerows(grid)

            print(f"{filename} done")
def norm_one():
    sums = 0
    csv_files = glob.glob('D:\\Sandro_Code\\channel_wise_velocity\\SLEEP_132_2_*_54_velocity.csv')
    for i, file in enumerate(csv_files):
        df = pd.read_csv(file, header=None)
        file_name = os.path.splitext(os.path.basename(file))[0]
        sums += df.sum().sum(axis=0)
#        if i + 1 < len(csv_files) and not similar(file_name, os.path.splitext(os.path.basename(csv_files[i+1]))[0]):
#            j += 1
    denominator = sums / (128**2 * 3)
    print(denominator)
    for i, file in enumerate(csv_files):
        df = pd.read_csv(file, header=None)
        print(file)
        file_name = os.path.splitext(os.path.basename(file))[0]
        df.div(denominator).to_csv(Path(f"{file}"), index = False, header = False)
    #        if i + 1 < len(csv_files) and not similar(file_name, os.path.splitext(os.path.basename(csv_files[i+1]))[0]):
    #            j += 1