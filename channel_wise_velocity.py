import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
import numpy as np
import glob
from num_waves import similar, data, states, mice

grid_size = 128
grid_average = [[np.zeros(3).tolist() for _ in range(grid_size)] for _ in range(grid_size)]
averages = pd.DataFrame(grid_average)
def individual_csvs():
    for file in data:
        filename = file.strip()
        df = pd.read_csv(Path(f"D:\\{filename}\\stage05_channel-wave_characterization\\channel-wise_measures.csv"))
        grid = pd.DataFrame([[np.zeros(3).tolist() for _ in range(grid_size)] for _ in range(grid_size)])
        for index, row in df.iterrows():
            x = int(row['x_coords'])
            y = int(row['y_coords'])
            velocity = float(row['velocity_local'])
            x_direction = float(row['direction_local_x'])
            y_direction = float(row['direction_local_y'])
            grid.iloc[y,x] = [velocity, x_direction, y_direction]
        grid.to_csv(Path(f"D:\\Sandro_Code\\channel_wise_velocity\\{filename}_velocity.csv"), index = False, header=False, mode='w+')
def average_csv():
    csv_files = glob.glob('D:\\Sandro_Code\\channel_wise_velocity\\*velocity.csv')
    for row in range(grid_size):
        for column in range(grid_size):
            
            count = [len(data)] * 3

            for file in csv_files:

                df = pd.read_csv(file)
                if row < df.shape[0] and column < df.shape[1]:
                    cell_value = df.iloc[row, column]
                    cell_value = cell_value.replace('[', '')
                    cell_value = cell_value.replace(']', '') 
                    cell_value = cell_value.split(', ')
                    print(cell_value)
                for index, item in enumerate(cell_value):
                    
                    if item != 0:
                        averages.iloc[row, column][index] += float(cell_value[index])
                    else:
                        count[index] -= 1
            for index, item in enumerate(averages.iloc[row, column]):
                item /= count[index]
    averages.to_csv(Path(f"D:\\Sandro_Code\\channel_wise_velocity\\average_velocity-direction.csv"), index = False, header = False, mode='w+')
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