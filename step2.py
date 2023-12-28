import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.colors as mcolors
from difflib import SequenceMatcher
import glob
import csv
from step1 import states, mice, similar, divide, add, Heat_Mapper, check_mice
import os
from scipy.stats import gaussian_kde
import seaborn as sns
grid_size = 128

def Velocity_Violin(currdir, norm=False, avg=False):
    if norm and avg:
        print("No velocity violins produced. Please choose either norm or avg.")
        return
    if norm:
        print(f"Graphing normalized velocity violins...")
        csv_files = glob.glob(f'{currdir}\\*velocity_N.csv')
    elif avg:
        print(f"Graphing averaged velocity violins...")
        csv_files = glob.glob(f'{currdir}\\*v-avg*.csv')
    else:
        print(f"Graphing velocity violins...")
        csv_files = glob.glob(f'{currdir}\\*velocity.csv')
        
    v_list = np.empty(grid_size**2)
    flagType = 'velocity.csv'
    csv_files = glob.glob(f'{currdir}\\*{flagType}')
    for index, file in enumerate(csv_files):
        df = pd.read_csv(file, header=None)
        file_name = os.path.splitext(os.path.basename(file))[0].replace('velocity','')
        i = 0
        for row_index, row in df.iterrows():
            for col_index, col in enumerate(row):
                if i < len(v_list):
                    v_list[i] = row[col_index]
                i += 1
        print(f"Mouse: {file_name}, Actual Length: {i}")
        fig, ax = plt.subplots()
        kde = gaussian_kde(v_list)
        scaling_factor = 1 / np.max(kde(v_list))
        sns.violinplot(data=[v_list * scaling_factor], ax=ax, bw='scott')

        fig.suptitle(f"{file_name} Velocity Density Distribution", fontsize=15)
        ax.set_ylabel("log10 channel velocity [mm/s]")
        print(v_list)
        plt.savefig(os.path.join(currdir, f'{file_name}_violin.png'))

def Normalize(args, currdir):
    print("Normalizing CSVs...")
    flagType = 'velocity_N.csv'
    csv_files = glob.glob(f'{currdir}\\*{flagType}')
    mouse_n = 0
    sums = [0] * int(args.n)
    for i, file in enumerate(csv_files):
        df = pd.read_csv(file, header=None)
        file_name = os.path.splitext(os.path.basename(file))[0]
        sums[mouse_n] += df.sum().sum(axis=0)
        if i + 1 < len(csv_files):
            if not similar(file_name, os.path.splitext(os.path.basename(csv_files[i+1]))[0]):
                #print(mouse_n)
                mouse_n += 1
    denominator = [x / (128**2 * 3) for x in sums]
    mouse_n = 0
    for i, file in enumerate(csv_files):
        df = pd.read_csv(file, header=None)
        file_name = os.path.splitext(os.path.basename(file))[0]
        df = df.div(denominator[mouse_n])
        df.to_csv(Path(file), index = False, header = False)
        if i + 1 < len(csv_files) and not similar(file_name, os.path.splitext(os.path.basename(csv_files[i+1]))[0]):
            j += 1

def Average_CSVs(args, currdir):
    grid_average = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
    averages_nrem, averages_rem, averages_wake = pd.DataFrame(grid_average), pd.DataFrame(grid_average), pd.DataFrame(grid_average)
    flagTypeNorm = 'velocity_N.csv'
    norm_path = f'{currdir}\\*{flagTypeNorm}'
    flagTypeAvg = 'velocity.csv'
    avg_path = f'{currdir}\\*{flagTypeAvg}'
    #args.norm
    if args.norm:
        print("Producing average norm CSVs...")
        nrem_csv_files = [x for x in glob.glob(norm_path) if 'NREM' in x]
        rem_csv_files = [x for x in glob.glob(norm_path) if 'REM' in x]
        wake_csv_files = [x for x in glob.glob(norm_path) if 'WAKE' in x]
    #if not norm
    else:
        print("Producing average CSVs...")
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

def Num_Waves_OG( args, currdir):
    #0 -> Wake, 1 -> NREM, 2 -> REM
    print(f"Graphing number of waves...")
    table = []
    for i in range(int(args.n)):
        table.append([])
        for j in range(3):
            table[i].append(0)
    flagType = 'velocity.csv'
    csv_files = glob.glob(f'{currdir}\\*{flagType}')
    for i, file in enumerate(csv_files):
        file_name = os.path.splitext(os.path.basename(file))[0].replace('_velocity','')
        df = pd.read_csv(Path(f"{path_head}\\stage05_channel-wave_characterization\\velocity_local\\wavefronts_velocity_local.csv"), usecols=['wavefronts_id'])
        index = df.tail(1).index.item() #grabs last index's value in file
        num = df.at[index, 'wavefronts_id']
        if 'WAKE' in str(file_name):
            table[j][0] = num
        elif 'NREM' in str(file_name):
            table[j][1] = num
        elif 'REM' in str(file_name):
            table[j][2] = num
        if i + 1 < len(csv_files) and not similar(file_name, os.path.splitext(os.path.basename(csv_files[i+1]))[0]):
            j += 1
    df = pd.DataFrame({'States': states})
    mice_unique = []
    [mice_unique.append(item) for item in mice if item not in mice_unique]
    for index, column in enumerate(table):
        df.insert(index, mice_unique[index], table[index], True)
    state = df.pop('States')
    df.insert(0, state.name, state)
    df.to_csv(Path(f"{currdir}\\number_of_waves.csv"), index = False, mode='w+')

    #individual bar graphs
    for mouse in mice_unique:
        y = ['WAKE', 'NREM', 'REM']
        x = df[mouse]
        bgraph = plt.bar(y,x)
        bgraph[0].set_color('red')
        bgraph[1].set_color('blue')
        bgraph[2].set_color('green')
        plt.ylabel('number of waves')
        plt.title(mouse)
        plt.savefig(f"{currdir}\\{mouse}_number_of_waves.png")
    plt.clf()

    #line plot
    for mouse in mice_unique:
        y = ['WAKE', 'NREM', 'REM']
        x = df[mouse]
        plt.plot(y, x, label=mouse, marker='o', linewidth=2)
    plt.legend()
    plt.title('Waves Across States Comparison')
    plt.savefig(f"{currdir}\\number_of_waves_comparison.png")
    plt.clf()

    #total bar graph
    df.pop('States')
    df_total = df.sum(axis = 1) #sum up rows of table
    y = ['WAKE', 'NREM', 'REM']
    x = df_total
    bgraph= plt.bar(y,x)
    bgraph[0].set_color('red')
    bgraph[1].set_color('blue')
    bgraph[2].set_color('green')
    plt.title('Total Number of Waves')
    plt.xlabel('number of waves')
    plt.savefig(f"{currdir}\\total_number_of_waves.png")
    plt.clf()

def Num_Waves_Comp(currdir):
    print(f"Graphing number of waves...")
    dict = {}
    flagType = '_numwaves'
    csv_files = glob.glob(f'{currdir}\\*{flagType}.csv')
    for file in csv_files:
        file_name = os.path.splitext(os.path.basename(file))[0]
        file_name_nonum = file_name.replace(flagType,'')
        file_name_nostate = file_name_nonum.replace('WAKE','').replace('NREM','').replace('REM','')
        #print(file_name_nostate)
        with open(Path(f"{currdir}\\{file_name}.csv"), 'r') as f:
            csv_reader = csv.reader(f)
            data = next(csv_reader)
        #print(data[0])
        if file_name_nostate not in dict:
            dict[file_name_nostate] = [0,0,0]
        if 'WAKE' in str(file_name):
            dict[file_name_nostate][0] = float(data[0])
        elif 'NREM' in str(file_name):
            dict[file_name_nostate][1] = float(data[0])
        elif 'REM' in str(file_name):
            dict[file_name_nostate][2] = float(data[0])
    #print(dict)
    df = pd.DataFrame.from_dict(dict, orient='index', columns=states)
    df.to_csv(Path(f"{currdir}\\numwaves_comp.csv"), index = True, mode='w+')
    
    #individual bar graphs
    for mouse in dict.keys():
        x = df.loc[mouse]
        bgraph = plt.bar(states,x)
        bgraph[0].set_color('red')
        bgraph[1].set_color('blue')
        bgraph[2].set_color('green')
        plt.ylabel('Planarity')
        plt.title(mouse)
        plt.savefig(f"{currdir}\\{mouse}_numwaves.png")
    plt.clf()
    
    #line plot
    for mouse in dict.keys():
        x = df.loc[mouse]
        x_masked = np.where(x == 0, np.nan, x)
        plt.plot(states, x_masked, label=mouse, marker='o', linewidth=2)
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')
    plt.title('Number of Waves Across States Comparison')
    plt.tight_layout()
    plt.savefig(f'{currdir}\\numwaves_comp.png')
    plt.clf()

    #total waves per state bar graph
    df_total = df.sum(axis = 0) #sum up rows of table
    x = df_total
    bgraph= plt.bar(states,x)
    bgraph[0].set_color('red')
    bgraph[1].set_color('blue')
    bgraph[2].set_color('green')
    plt.title('Total Number of Waves')
    plt.xlabel('number of waves')
    plt.savefig(f"{currdir}\\numwaves_total.png")
    plt.clf()

def Avg_Polar(data_path, args, data, currdir):
    # Master list containing every x and y coord based on parameter/argument
    all_directionY = []
    all_directionX = []

    # Lists for normalized averages after taking individual averages
    avg_x_normalized = []
    avg_y_normalized = []

    all_files = [entry['filename'] for entry in data]
    print(f"Graphing polar histogram for entries: {all_files}...")
    
    for entry in data:
        filename = entry['filename']
        wave_ids = entry['wave_ids']
        df = pd.read_csv(Path(f"{data_path}{filename}\\stage05_channel-wave_characterization\\direction_local\\wavefronts_direction_local.csv"))

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

    # Create a polar histogram each method
    fig, (ax, ax1) = plt.subplots(1,2, subplot_kw={'projection': 'polar'})
    ax.hist(angles, bins=36, range=(-np.pi, np.pi), density=True)
    ax.set_xlabel('normalized by vector length')

    ax1.hist(angles_norm, bins=36, range=(-np.pi, np.pi), density=True)
    ax1.set_xlabel('normalized by # vectors in wave')
    
    # Plot the weighted average line
    ax.plot([0, weighted_average_angle], [0, ax.get_ylim()[1]], color='red', linewidth=2)
    ax1.plot([0, weighted_average_angle1], [0, ax1.get_ylim()[1]], color='black', linewidth=2)

    setOutputName = "avg_polar"
    fig.suptitle(f"avg for {args.n} {'mouse' if args.n == '1' else 'mice'}", fontsize=16)
    fig.savefig(os.path.join(currdir, f'{setOutputName}.png'))
    plt.close()

def Planarity_Comp(currdir):
    print(f"Graphing planarity...")
    dict = {}
    flagType = '_planarity'
    csv_files = glob.glob(f'{currdir}\\*{flagType}.csv')
    for file in csv_files:
        file_name = os.path.splitext(os.path.basename(file))[0]
        file_name_noplanarity = file_name.replace(flagType,'')
        #if 'WAKE' in str(file_name_noplanarity) or 'NREM' in str(file_name_noplanarity) or 'REM' in str(file_name_noplanarity):
        file_name_nostate = file_name_noplanarity.replace('WAKE','').replace('NREM','').replace('REM','')
        #print(file_name_nostate)
        with open(Path(f"{currdir}\\{file_name}.csv"), 'r') as f:
            csv_reader = csv.reader(f)
            data = next(csv_reader)
        #print(data[0])
        if file_name_nostate not in dict:
            dict[file_name_nostate] = [0,0,0]
        if 'WAKE' in str(file_name):
            dict[file_name_nostate][0] = float(data[0])
        elif 'NREM' in str(file_name):
            dict[file_name_nostate][1] = float(data[0])
        elif 'REM' in str(file_name):
            dict[file_name_nostate][2] = float(data[0])
    #print(dict)
    df = pd.DataFrame.from_dict(dict, orient='index', columns=states)
    df.to_csv(Path(f"{currdir}\\planarity_comp.csv"), index = True, mode='w+')
    #individual bar graphs
    for mouse in dict.keys():
        x = df.loc[mouse]
        bgraph = plt.bar(states,x)
        bgraph[0].set_color('red')
        bgraph[1].set_color('blue')
        bgraph[2].set_color('green')
        plt.ylabel('Planarity')
        plt.title(mouse)
        plt.savefig(f"{currdir}\\{mouse}_planarity.png")
    plt.clf()
    
    #line plot
    for mouse in dict.keys():
        x = df.loc[mouse]
        x_masked = np.where(x == 0, np.nan, x)
        plt.plot(states, x_masked, label=mouse, marker='o', linewidth=2)
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')
    plt.title('Planarity Across States Comparison')
    plt.tight_layout()
    plt.savefig(f'{currdir}\\planarity_comp.png')
    plt.clf()

def Heat_Mapper(currdir, norm=False, avg=False):
    if norm and avg:
        print("No heatmaps produced. Please choose either norm or avg.")
        return
    if norm:
        print(f"Producing normalized heatmaps...")
        csv_files = glob.glob(f'{currdir}\\*velocity_N.csv')
        scale_min = 0
        scale_max = 2
    elif avg:
        print(f"Producing average heatmaps...")
        csv_files = glob.glob(f'{currdir}\\*v-avg*.csv')
        scale_min = 0
        scale_max = 2
    else:
        print(f"Producing velocity heatmaps...")
        csv_files = glob.glob(f'{currdir}\\*velocity.csv')
        scale_min = 43518.26345849037
        scale_max = 2272431268.2241783
    
    for file in csv_files:
        df = pd.read_csv(file, header=None)
        file_name = os.path.splitext(os.path.basename(file))[0]
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
        ax.set_title(file_name)
        plt.savefig(os.path.join(currdir, f'{file_name}_heatmap.png'))
        plt.close()

def run(data_path, args, data, currdir):
    Avg_Polar(data_path, args, data, currdir)
    if args.v:
        Velocity_Violin(currdir)
    if args.norm:
        Normalize(args, currdir)
    if args.avg:
        Average_CSVs(args, currdir)
    if args.p:
        Planarity_Comp(currdir)
    if args.num:
        Num_Waves_Comp(currdir)
    Heat_Mapper(currdir, args.norm, args.avg)
