import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
<<<<<<< HEAD
import matplotlib.colors as mcolors
from difflib import SequenceMatcher
import glob
import csv
from num_waves import similar, data, states, mice_unique
import os
import argparse
=======
from matplotlib.colors import LinearSegmentedColormap
import h5py
from difflib import SequenceMatcher
import glob
from num_waves import similar, data, states, mice
>>>>>>> e6d7ac195490f31a2292f49d708da523ff121bd4

#parser = argparse.ArgumentParser(
#                    prog='Velocity Heat Maps',
#                    description='Plots slow-wave average velocity data per channel on a heat map',
#                    epilog='Hope this works')
#parser.add_argument("--avg", default=False, action="store_true",
#                    help="Flag if you want to make average csv")
#parser.add_argument('wave_ids', type=int, metavar='N', nargs='*', 
#                    help='input desired wavefronts (default is all). Delimit ranges by commas: w-x,y,z-a')
#args = parser.parse_args()

grid_size = 128
<<<<<<< HEAD
#wave_ids = args.wave_ids.split(',')
=======
grid_average = [[np.zeros(3).tolist() for _ in range(grid_size)] for _ in range(grid_size)]
averages = pd.DataFrame(grid_average)
>>>>>>> e6d7ac195490f31a2292f49d708da523ff121bd4
def individual_csvs():
    for file in data:
        filename = file.strip()
        df = pd.read_csv(Path(f"D:\\{filename}\\stage05_channel-wave_characterization\\channel-wise_measures.csv"))
        grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
        max_wave = np.max(df.wavefronts_id)
        channel_wave_count = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
        for index, row in df.iterrows():
            x = int(row['x_coords'])
            y = int(row['y_coords'])
<<<<<<< HEAD
            velocity = float(row['velocity_local'])      
            grid[y][x] += velocity
            channel_wave_count[y][x] += 1   
        for index, row in enumerate(grid):
            for col_index, col in enumerate(row):
                if channel_wave_count[index][col_index] != 0:
                    grid[index][col_index] /= channel_wave_count[index][col_index]
        print(channel_wave_count)
        with open(Path(f"D:\\Sandro_Code\\channel_wise_velocity\\{filename}_velocity.csv"), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(grid)
        
        print(f"{filename} done")
=======
            velocity = float(row['velocity_local'])
            x_direction = float(row['direction_local_x'])
            y_direction = float(row['direction_local_y'])
            grid.iloc[y,x] = [velocity, x_direction, y_direction]
        grid.to_csv(Path(f"D:\\Sandro_Code\\channel_wise_velocity\\{filename}_velocity.csv"), index = False, header=False, mode='w+')
>>>>>>> e6d7ac195490f31a2292f49d708da523ff121bd4
def average_csv():
    grid_average = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
    averages_nrem, averages_rem, averages_wake = pd.DataFrame(grid_average), pd.DataFrame(grid_average), pd.DataFrame(grid_average)
    nrem_csv_files = [x for x in glob.glob('D:\\Sandro_Code\\channel_wise_velocity\\*velocity.csv') if 'NREM' in x]
    rem_csv_files = [x for x in glob.glob('D:\\Sandro_Code\\channel_wise_velocity\\*velocity.csv') if 'REM' in x]
    wake_csv_files = [x for x in glob.glob('D:\\Sandro_Code\\channel_wise_velocity\\*velocity.csv') if 'WAKE' in x]
    for row in range(grid_size):
        for column in range(grid_size):

            add(nrem_csv_files, row, column, averages_nrem)
            add(rem_csv_files, row, column, averages_rem)
            add(wake_csv_files, row, column, averages_wake)

    divide(averages_nrem, len(nrem_csv_files)).to_csv(Path(f"D:\\Sandro_Code\\channel_wise_velocity\\v_average_nrem.csv"), index = False, header = False, mode='w+')
    divide(averages_rem, len(rem_csv_files)).to_csv(Path(f"D:\\Sandro_Code\\channel_wise_velocity\\v_average_rem.csv"), index = False, header = False, mode='w+')
    divide(averages_wake, len(wake_csv_files)).to_csv(Path(f"D:\\Sandro_Code\\channel_wise_velocity\\v_average_wake.csv"), index = False, header = False, mode='w+')
def specific_wavefront():
    filename = "D:\\ANESTH_EricMs1Ket_KX_16\\stage05_channel-wave_characterization\\channel-wise_measures.csv"
    df = pd.read_csv(Path(filename))
    grid = pd.DataFrame([[np.zeros(3).tolist() for _ in range(grid_size)] for _ in range(grid_size)])
    for index, row in df.iterrows():
        if row['wavefronts_id'] == 7:
            x = int(row['x_coords'])
            y = int(row['y_coords'])
            velocity = float(row['velocity_local'])
            x_direction = float(row['direction_local_x'])
            y_direction = float(row['direction_local_y'])
            grid.iloc[y,x] = [velocity, x_direction, y_direction]
    grid.to_csv(Path(f"D:\\Sandro_Code\\channel_wise_velocity\\wavefronts_test_ket_1.csv"), index = False, header=False, mode='w+')
#for a individual mouse, need to modify because there are multiple values per position
def heat_quiver():
    #set up empty arrays
    #csv_files = glob.glob('D:\\Sandro_Code\\channel_wise_velocity\\v_average*.csv')
    csv_files = glob.glob('D:\\Sandro_Code\\channel_wise_velocity\\*velocity.csv')
    for file in csv_files:
        df = pd.read_csv(file, header=None)
        file_name = os.path.basename(file)
        num_rows = len(df)
        num_columns = len(df.columns)

        x = np.array([i for i in range(num_columns)])
        y = np.array([i for i in range(num_rows)])
        row_shape, col_shape = df.shape
        heat_data = np.zeros((row_shape, col_shape))

<<<<<<< HEAD
        #retrieve data from csv
        for row_index, row in df.iterrows():
            for col_index, col in enumerate(row):
                heat_data[row_index, col_index] = float(row[col_index])
    #    vector_length = 1
    #    lengths = np.sqrt(x_dir**2 + y_dir**2)
    #    filter = (lengths != 0) & np.isfinite(lengths)
    #    np.seterr(divide='ignore', invalid='ignore')
    #    U = np.where(filter, x_dir * vector_length / lengths, 0)
    #    V = np.where(filter, y_dir * vector_length / lengths, 0)
    #    np.seterr(divide='warn', invalid='warn')

        #mask 0 data
    #    filter = (U != 0) & (V != 0)
    #    U = np.where(filter, U, np.nan)
    #    V = np.where(filter, V, np.nan)
    #    filter = (heat_data != 0)
    #    heat_data = np.where(filter, heat_data, np.nan)

        #set scale
        #scale_min = np.nanmean(heat_data) - np.nanstd(heat_data)
        #scale_max = np.nanmean(heat_data) + np.nanstd(heat_data)
        scale_min = 43518.26345849037
        scale_max = 2272431268.2241783
        fig, ax = plt.subplots()

        #quiverplot
    #vectorfield = ax.quiver(X, Y, U, V, scale=100, headlength=4, headaxislength=3, minshaft = 1.5, pivot='middle')
        cmap_nonzero = plt.get_cmap('rainbow')
        cmap_custom = mcolors.ListedColormap(['white'] + [cmap_nonzero(i) for i in range(1, cmap_nonzero.N)])
        norm = mcolors.Normalize(vmin= scale_min, vmax=scale_max)
        heatmap = ax.imshow(heat_data, cmap=cmap_custom, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], norm=norm)
        ax.set_frame_on(False)
        fig.colorbar(heatmap)
        ax.set_aspect('equal')
        ax.set_title(f'{os.path.splitext(file_name)[0]}')
        plt.savefig(Path(f"D:\\Sandro_Code\\channel_wise_velocity\\{os.path.splitext(file_name)[0]}.png"))
        plt.close()
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

#individual_csvs()
#if args.avg:
#    average_csv()
=======
    x = np.array([i for i in range(num_columns)])
    y = np.array([i for i in range(num_rows)])
    X,Y  = np.meshgrid(x,y)
    row_shape, col_shape = df.shape
    x_dir = np.zeros((row_shape, col_shape))
    y_dir = np.zeros((row_shape, col_shape))
    heat_data = np.zeros((row_shape, col_shape))

    #retrieve data from csv
    for row_index, row in df.iterrows():
        for col_index, col in enumerate(row):
            cell_value = row[col_index]
            cell_value = cell_value.replace('[', '')
            cell_value = cell_value.replace(']', '')
            cell_value = cell_value.split(', ')
            heat_data[row_index, col_index] = float(cell_value[0])
            x_dir[row_index, col_index] = float(cell_value[1])
            y_dir[row_index, col_index] = float(cell_value[2])

    #normalize vector magnitude
    vector_length = 1
    lengths = np.sqrt(x_dir**2 + y_dir**2)
    filter = (lengths != 0) & np.isfinite(lengths)
    np.seterr(divide='ignore', invalid='ignore')
    U = np.where(filter, x_dir * vector_length / lengths, 0)
    V = np.where(filter, y_dir * vector_length / lengths, 0)
    np.seterr(divide='warn', invalid='warn')

    #mask 0 data
    filter = (U != 0) & (V != 0)
    U = np.where(filter, U, np.nan)
    V = np.where(filter, V, np.nan)
    filter = (heat_data != 0)
    heat_data = np.where(filter, heat_data, np.nan)

    #mask brain sides
    load_mask = h5py.File('C:\\Users\\sandro\\Downloads\\week0allmask.mat', 'r')
    mask = load_mask.get('papermask2')
    mask = np.array(mask)
    masked_heat_data = heat_data * mask

    #set scale
    scale_min = np.nanmean(masked_heat_data) - np.nanstd(masked_heat_data)
    scale_max = np.nanmean(masked_heat_data) + np.nanstd(masked_heat_data)

    fig, ax = plt.subplots()

    #quiverplot
    vectorfield = ax.quiver(X, Y, U, V, scale=100, headlength=4, headaxislength=3, minshaft = 1.5, pivot='middle')

    #heatmap
    colors = ['blue', 'yellow']
    cmap = LinearSegmentedColormap.from_list('CustomColormap', colors)
    heatmap = ax.imshow(masked_heat_data, cmap=cmap, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], vmin= scale_min, vmax = scale_max)
    ax.set_frame_on(False)
    fig.colorbar(heatmap)
    ax.set_aspect('equal')
    plt.savefig(Path("D:\\Sandro_Code\\channel_wise_velocity\\wavefronts_test_ket.pdf"))
def velocity_violin():
    dict = {
        'WAKE': np.empty(grid_size**2),
        'NREM': np.empty(grid_size**2),
        'REM': np.empty(grid_size**2),
    }
    for index, mouse in enumerate(data):
        df = pd.read_csv(f"D:\\Sandro_Code\\channel_wise_velocity\\{mouse}_velocity.csv")
        if 'WAKE' in mouse:
            length = len(dict['WAKE'])
            print(f"Mouse: {mouse}, Expected Length: {length}")
        if 'NREM' in mouse:
            length = len(dict['NREM'])
            print(f"Mouse: {mouse}, Expected Length: {length}")
        if 'REM' in mouse:
            length = len(dict['REM'])
            print(f"Mouse: {mouse}, Expected Length: {length}")
        i = 0
        for row_index, row in df.iterrows():
            for col_index, col in enumerate(row):
                cell_value = row[col_index]
                cell_value = cell_value.replace('[', '')
                cell_value = cell_value.replace(']', '')
                cell_value = cell_value.split(', ')
                value = float(cell_value[0])
                if value <= 0:
                    for key, array in dict.items():
                        if key in mouse and i < length:
                            array[i] = np.nan
                else:
                    for key, array in dict.items():
                        if key in mouse and i < length:
                            array[i] = value
                i += 1
        print(f"Mouse: {mouse}, Actual Length: {i}")
        if not similar(data[index], data[index+1]):
            for key, array in dict.items():
                dict[key] = np.log10(array[np.isfinite(array)])
                
            fig, axes = plt.subplots(1, 3)

            #wake
            axes[0].violinplot(dict['WAKE'], showmedians=True)
            #nrem
            axes[1].violinplot(dict['NREM'], showmedians=True)
            #rem
            axes[2].violinplot(dict['REM'], showmedians=True)

            plt.show()
>>>>>>> e6d7ac195490f31a2292f49d708da523ff121bd4
heat_quiver()