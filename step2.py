import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.colors as mcolors
from difflib import SequenceMatcher
import glob
import csv
from step1 import states, mice, similar, divide, add, check_mice
import os
from scipy.stats import gaussian_kde
import seaborn as sns
grid_size = 128
'''
def Velocity_Violin(currdir, norm=False, avg=False):
    if norm and avg:
        print("No velocity violins produced. Please choose either norm or avg.")
        return
    if norm:
        print(f"Graphing normalized velocity violins...")
        flagType = 'velocity_N.csv'
        csv_files = glob.glob(f'{currdir}\\*velocity_N.csv')
    elif avg:
        print(f"Graphing averaged velocity violins...")
        flagType = 'v-avg.csv'
        csv_files = glob.glob(f'{currdir}\\*v-avg*.csv')
    else:
        print(f"Graphing velocity violins...")
        flagType = 'velocity.csv'
        csv_files = glob.glob(f'{currdir}\\*velocity.csv')
        
    v_list = np.empty(grid_size**2)
    for index, file in enumerate(csv_files):
        df = pd.read_csv(file, header=None)
        file_name = os.path.splitext(os.path.basename(file))[0].replace(flagType,'')
        print(file_name)
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
        sns.violinplot(data=[v_list * scaling_factor], ax=ax, bw='scott', showfliers=False)

        fig.suptitle(f"{file_name} Velocity Density Distribution", fontsize=15)
        ax.set_ylabel("log10 channel velocity [mm/s]")
        plt.savefig(os.path.join(currdir, f'{file_name}_violin.png'))
'''
def Velocity_Violin(currdir, norm=False, avg=False):
    if norm:
        print(f"Graphing normalized velocity violins...")
        flagType = 'velocity_N'
        csv_files = glob.glob(f'{currdir}\\*velocity_N.csv')
    elif avg:
        print(f"Graphing averaged velocity violins...")
        flagType = 'v-avg'
        csv_files = glob.glob(f'{currdir}\\*v-avg*.csv')
    else:
        print(f"Graphing velocity violins...")
        flagType = 'velocity'
        csv_files = glob.glob(f'{currdir}\\*velocity.csv')

    print("current directory: ", currdir)
    print("csv_files: ", csv_files)
    
    for index, file in enumerate(csv_files):
        df = pd.read_csv(file, header=None)
        file_name = os.path.splitext(os.path.basename(file))[0].replace(flagType,'')
        print(file_name)

        v_list = df.values.flatten()
        Q1 = np.percentile(v_list, 25)
        Q3 = np.percentile(v_list, 75)
        IQR = Q3 - Q1
        v_list_no_outliers = v_list[(v_list >= Q1 - 1.5 * IQR) & (v_list <= Q3 + 1.5 * IQR)]

        print(f"Mouse: {file_name}, Original Length: {len(v_list)}, Length after removing outliers: {len(v_list_no_outliers)}")

        fig, ax = plt.subplots()

        kde = gaussian_kde(v_list_no_outliers)
        scaling_factor = 1 / np.max(kde(v_list_no_outliers))

        sns.violinplot(data=[v_list_no_outliers * scaling_factor], ax=ax, bw='scott')

        fig.suptitle(f"{file_name} Velocity Density Distribution", fontsize=15)
        ax.set_ylabel("log10 channel velocity [mm/s]")
        plt.savefig(os.path.join(currdir, f'{file_name}_violin.png'))
        plt.close()

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
                mouse_n += 1
    denominator = [x / (128**2 * 3) for x in sums]
    mouse_n = 0
    for i, file in enumerate(csv_files):
        df = pd.read_csv(file, header=None)
        file_name = os.path.splitext(os.path.basename(file))[0]
        df = df.div(denominator[mouse_n])
        df.to_csv(Path(file), index = False, header = False)
        if i + 1 < len(csv_files) and not similar(file_name, os.path.splitext(os.path.basename(csv_files[i+1]))[0]):
            mouse_n += 1

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
        rem_csv_files = [x for x in glob.glob(norm_path) if 'REM' in x and 'NREM' not in x]
        wake_csv_files = [x for x in glob.glob(norm_path) if 'WAKE' in x]
    #if not norm
    else:
        print("Producing average CSVs...")
        nrem_csv_files = [x for x in glob.glob(avg_path) if 'NREM' in x]
        rem_csv_files = [x for x in glob.glob(avg_path) if 'REM' in x and 'NREM' not in x]
        wake_csv_files = [x for x in glob.glob(avg_path) if 'WAKE' in x]

    for row in range(grid_size):
        for column in range(grid_size):
            add(nrem_csv_files, row, column, averages_nrem)
            add(rem_csv_files, row, column, averages_rem)
            add(wake_csv_files, row, column, averages_wake)
    if len(nrem_csv_files) > 0:
        nrem_avg_path = norm_path.replace('*velocity', 'NREM_v-avg')
        divide(averages_nrem, len(nrem_csv_files)).to_csv(Path(nrem_avg_path), index = False, header = False, mode='w+')
    if len(rem_csv_files) > 0:
        rem_avg_path = norm_path.replace('*velocity', 'REM_v-avg')
        divide(averages_rem, len(rem_csv_files)).to_csv(Path(rem_avg_path), index = False, header = False, mode='w+')
    if len(wake_csv_files) > 0:
        wake_avg_path = norm_path.replace('*velocity', 'WAKE_v-avg')
        divide(averages_wake, len(wake_csv_files)).to_csv(Path(wake_avg_path), index = False, header = False, mode='w+') 
   
def Num_Waves_Comp(currdir):
    print(f"Graphing number of waves...")
    dict = {}
    flagType = '_numwaves'
    csv_files = glob.glob(f'{currdir}\\*{flagType}.csv')
    for file in csv_files:
        file_name = os.path.splitext(os.path.basename(file))[0]
        file_name_nonum = file_name.replace(flagType,'')
        file_name_nostate = file_name_nonum.replace('WAKE','').replace('NREM','').replace('REM','')
        #with open(Path(f"{currdir}\\{file_name}.csv"), 'r') as f:
        with open(os.path.join(currdir, f'{file_name}.csv'), 'r') as f:
            csv_reader = csv.reader(f)
            data = next(csv_reader)
        if file_name_nostate not in dict:
            dict[file_name_nostate] = [0,0,0]
        if 'WAKE' in str(file_name):
            dict[file_name_nostate][0] = float(data[0])
        elif 'NREM' in str(file_name):
            dict[file_name_nostate][1] = float(data[0])
        elif 'REM' in str(file_name):
            dict[file_name_nostate][2] = float(data[0])
    df = pd.DataFrame.from_dict(dict, orient='index', columns=states)
    #df.to_csv(Path(f"{currdir}\\numwaves_comp.csv"), index = True, mode='w+')
    df.to_csv(os.path.join(currdir, 'numwaves_comp.csv'), index = True, mode='w+') #save to csv
    
    #individual bar graphs
    for mouse in dict.keys():
        x = df.loc[mouse]
        bgraph = plt.bar(states,x)
        bgraph[0].set_color('red')
        bgraph[1].set_color('blue')
        bgraph[2].set_color('green')
        plt.ylabel('Planarity')
        plt.title(mouse)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        #plt.savefig(f"{currdir}\\{mouse}_numwaves.png")
        plt.savefig(os.path.join(currdir, f'{mouse}_numwaves.png'))
        plt.close()
    plt.clf()
    
    #line plot
    for mouse in dict.keys():
        x = df.loc[mouse]
        x_masked = np.where(x == 0, np.nan, x)
        plt.plot(states, x_masked, label=mouse, marker='o', linewidth=2)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')
    plt.title('Number of Waves Across States Comparison')
    plt.tight_layout()
    #plt.savefig(f'{currdir}\\numwaves_comp.png')
    plt.savefig(os.path.join(currdir, 'numwaves_comp.png'))
    plt.close()
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
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    #plt.savefig(f"{currdir}\\numwaves_total.png")
    plt.savefig(os.path.join(currdir, 'numwaves_total.png'))
    plt.close()
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
    #iterate through each mouse
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
        file_name_nostate = file_name_noplanarity.replace('WAKE','').replace('NREM','').replace('REM','')
        #with open(Path(f"{currdir}\\{file_name}.csv"), 'r') as f:
        with open(os.path.join(currdir, f'{file_name}.csv'), 'r') as f:
            csv_reader = csv.reader(f)
            data = next(csv_reader)
        if file_name_nostate not in dict:
            dict[file_name_nostate] = [0,0,0]
        if 'WAKE' in str(file_name):
            dict[file_name_nostate][0] = float(data[0])
        elif 'NREM' in str(file_name):
            dict[file_name_nostate][1] = float(data[0])
        elif 'REM' in str(file_name):
            dict[file_name_nostate][2] = float(data[0])
        print("planarity dict", dict)
            
    df = pd.DataFrame.from_dict(dict, orient='index', columns=states)
    #df.to_csv(Path(f"{currdir}\\planarity_comp.csv"), index = True, mode='w+')
    df.to_csv(os.path.join(currdir, 'planarity_comp.csv'), index = True, mode='w+') #save to csv
    #individual bar graphs
    for mouse in dict.keys():
        x = df.loc[mouse]
        bgraph = plt.bar(states,x)
        bgraph[0].set_color('red')
        bgraph[1].set_color('blue')
        bgraph[2].set_color('green')
        plt.ylabel('Planarity')
        plt.title(mouse)
        #plt.savefig(f"{currdir}\\{mouse}_planarity.png")
        plt.savefig(os.path.join(currdir, f'{mouse}_planarity.png'))
        plt.close()
    plt.clf()
    
    #line plot
    for mouse in dict.keys():
        x = df.loc[mouse]
        x_masked = np.where(x == 0, np.nan, x)
        plt.plot(states, x_masked, label=mouse, marker='o', linewidth=2)
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')
    plt.title('Planarity Across States Comparison')
    plt.tight_layout()
    #plt.savefig(f'{currdir}\\planarity_comp.png')
    plt.savefig(os.path.join(currdir, 'planarity_comp.png'))
    plt.clf()

def Velocity_Topo_Average(currdir, norm=False, avg=False):
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
        file_name = os.path.splitext(os.path.basename(file))[0].replace("velocity",'').replace("velocity_N",'').replace("v-avg",'')
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
       
        plt.figure(figsize=(10, 8))
        heatmap = sns.heatmap(heat_data, 
                              annot=False, 
                              cmap= "viridis", robust=True, mask=(heat_data == 0), xticklabels=False, yticklabels=False, square=True, cbar=True)
        heatmap.tick_params(left=False, bottom=False)
        heatmap.set_aspect('equal')
        heatmap.invert_yaxis()
        heatmap.set_title(file_name, fontsize=18, loc="center")
        plt.savefig(os.path.join(currdir, f'{file_name}_heatmap.png'))

def Num_Waves_Topo_Cumulative(currdir):
    print(f"Graphing cumulative topo waves...")
    wake_list = np.zeros(grid_size**2)
    nrem_list = np.zeros(grid_size**2)
    rem_list = np.zeros(grid_size**2)
    count = [0] * 3 #wake, nrem, rem
    flagType = 'numwave_topo'
    #csv_files = glob.glob(f'D:\\Sandro_Code\\landsness_imaging\\output\\temp\\*{flagType}.csv')
    csv_files = glob.glob(f'{currdir}\\*{flagType}.csv') #maybe use a 3 pronged if statement to get rid of the need for the flagType variable
    for file in csv_files:
        channel_list = np.genfromtxt(file, delimiter=',')
        channel_list = channel_list.flatten()
        file_name = os.path.splitext(os.path.basename(file))[0]
        file_name_nolabel = file_name.replace(flagType,'')
        file_name_nostate = file_name_nolabel.replace('WAKE','').replace('NREM','').replace('REM','')
        for i, channel in enumerate(channel_list):
            if 'WAKE' in str(file_name):
                wake_list[i] += channel_list[i]
                count[0] += 1
            elif 'NREM' in str(file_name):
                nrem_list[i] += channel_list[i]
                count[1] += 1
            elif 'REM' in str(file_name):
                rem_list[i] += channel_list[i]
                count[2] += 1

    
        #create individual logic to avoid warnings, unwanted outputs, 
        wake_list = np.divide(wake_list, count[0])
        nrem_list = np.divide(nrem_list, count[1])
        rem_list = np.divide(rem_list, count[2])
        wake_grid = np.rot90(wake_list.reshape(-1, grid_size))
        nrem_grid = np.rot90(nrem_list.reshape(-1, grid_size))
        rem_grid = np.rot90(rem_list.reshape(-1, grid_size))

        #csv

       # with open(Path(f"{currdir}\\{file_name}.csv"), 'r') as f:

        np.savetxt(f"{currdir}\\wake_numwave_topo.csv", wake_grid, delimiter=',')
        #np.savetxt(f"D:\\Sandro_Code\\landsness_imaging\\output\\temp\\nrem_numwave_topo.csv", nrem_grid, delimiter=',')
        np.savetxt(f"{currdir}\\nrem_numwave_topo.csv", nrem_grid, delimiter=',')
        np.savetxt(f"{currdir}\\rem_numwave_topo.csv", rem_grid, delimiter=',')
        #np.savetxt(f".\\output\\{filename}_numwave_topo.csv", rotated_grid, delimiter=',')




        # heatmap
        plt.figure(figsize=(10, 8))
        heatmap = sns.heatmap(wake_grid, 
                                annot=False, 
                                cmap= "viridis", robust=True, xticklabels=False, yticklabels=False, square=True, cbar=True)
        heatmap.tick_params(left=False, bottom=False)
        heatmap.set_aspect('equal')
        heatmap.invert_yaxis()
        heatmap.set_title("Wake Topo Num Waves", fontsize=18, loc="center")
        #plt.show()
        #plt.savefig("D:\\Sandro_Code\\landsness_imaging\\output\\temp\\wake_numwave_topo.png")
        plt.savefig(os.path.join(currdir, 'wake_numwave_topo.png'))
        plt.close()

        plt.figure(figsize=(10, 8))
        heatmap = sns.heatmap(nrem_grid, 
                                annot=False, 
                                cmap= "viridis", robust=True, xticklabels=False, yticklabels=False, square=True, cbar=True)
        heatmap.tick_params(left=False, bottom=False)
        heatmap.set_aspect('equal')
        heatmap.invert_yaxis()
        heatmap.set_title("NREM Topo Num Waves", fontsize=18, loc="center")
        #plt.savefig("D:\\Sandro_Code\\landsness_imaging\\output\\temp\\nrem_numwave_topo.png")
        plt.savefig(os.path.join(currdir, 'nrem_numwave_topo.png'))
        plt.close()

        plt.figure(figsize=(10, 8))
        heatmap = sns.heatmap(rem_grid, 
                                annot=False, 
                                cmap= "viridis", robust=True, xticklabels=False, yticklabels=False, square=True, cbar=True)
        heatmap.tick_params(left=False, bottom=False)
        heatmap.set_aspect('equal')
        heatmap.invert_yaxis()
        heatmap.set_title("REM Topo Num Waves", fontsize=18, loc="center")
        #plt.savefig("D:\\Sandro_Code\\landsness_imaging\\output\\temp\\rem_numwave_topo.png")
        plt.savefig(os.path.join(currdir, 'rem_numwave_topo.png'))
        plt.close()

def run(data_path, args, data, currdir):
    Avg_Polar(data_path, args, data, currdir) #averge polar histogram across all mice
    #if args.norm:
        #Normalize(args, currdir)
    #if args.avg:
        #Average_CSVs(args, currdir)
    if args.v:
        Velocity_Violin(currdir, args.norm, args.avg) 
    if args.p:
        Planarity_Comp(currdir)
    if args.num:
        Num_Waves_Comp(currdir) #number of waves per state per mice
    Velocity_Topo_Average(currdir, args.norm, args.avg) #was previously in Heat_Mapper(currdir, args.norm, args.avg)
    Num_Waves_Topo_Cumulative(currdir)
