import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob
import csv
import os
from scipy.stats import gaussian_kde
import seaborn as sns
grid_size = 128

#helper function
def state_identifier(file, state_list):
    file_name_split = file.split('_')
    for name_part in file_name_split:
        for state in state_list:
            if name_part == state:
                return state

#visualization functions
def Velocity_Violin(args, currdir):
    #group csvs depending if norm flag is called
    if args.norm:
        print(f"Entered Velocity_Violin with Norm Flag")
        flagType = 'velocity_norm'
        csv_files = glob.glob(f'{currdir}\\*velocity_norm.csv')
    else:
        print(f"Entered Velocity_Violin")
        flagType = 'velocity'
        csv_files = glob.glob(f'{currdir}\\*velocity.csv')

    #iterate through all csvs
    for file in csv_files:
        df = pd.read_csv(file, header=None)
        file_name = os.path.splitext(os.path.basename(file))[0].replace(flagType,'')
        
        #remove outliers based on quartiles
        v_list = df.values.flatten()
        Q1 = np.percentile(v_list, 25)
        Q3 = np.percentile(v_list, 75)
        IQR = Q3 - Q1
        v_list_no_outliers = v_list[(v_list >= Q1 - 1.5 * IQR) & (v_list <= Q3 + 1.5 * IQR)]

        #create figure
        fig, ax = plt.subplots()
        
        #scaling
        kde = gaussian_kde(v_list_no_outliers)
        scaling_factor = 1 / np.max(kde(v_list_no_outliers))

        #violin plot parameters
        sns.violinplot(data=[v_list_no_outliers * scaling_factor], ax=ax, bw_method='scott')
        abbreviated_title = file_name.replace('velocity_norm','').replace('velocity','')
        fig.suptitle(f"{abbreviated_title} velocity density distribution", fontsize=15)
        ax.set_ylabel("log10 channel velocity [mm/s]")
        ax.set_xlabel("Mouse")

        plt.savefig(os.path.join(currdir, f'{file_name}violin.png'))
        plt.close()

    average_files = glob.glob(f'{currdir}\\*average.csv')
    #iterate through average csvs
    for file in average_files:
        df = pd.read_csv(file, header=None)
        file_name = os.path.splitext(os.path.basename(file))[0]
        
        #remove outliers based on quartiles
        v_list = df.values.flatten()
        Q1 = np.percentile(v_list, 25)
        Q3 = np.percentile(v_list, 75)
        IQR = Q3 - Q1
        v_list_no_outliers = v_list[(v_list >= Q1 - 1.5 * IQR) & (v_list <= Q3 + 1.5 * IQR)]

        #create figure
        fig, ax = plt.subplots()
        
        #scaling
        kde = gaussian_kde(v_list_no_outliers)
        scaling_factor = 1 / np.max(kde(v_list_no_outliers))

        #violin plot parameters
        sns.violinplot(data=[v_list_no_outliers * scaling_factor], ax=ax, bw_method='scott')
        abbreviated_title = file_name.replace('velocity_norm','').replace('velocity','')
        fig.suptitle(f"{abbreviated_title} velocity density distribution", fontsize=15)
        ax.set_ylabel("log10 channel velocity [mm/s]")
        ax.set_xlabel("Mouse")

        plt.savefig(os.path.join(currdir, f'{file_name}_violin.png'))
        plt.close()

def Normalize(currdir):
    print("Entered Normalize")

    #group csvs, initialize means list for each csv
    flagType = 'velocity.csv'
    csv_files = glob.glob(f'{currdir}\\*{flagType}')
    means = [0] * len(csv_files)

    #iterate through all csvs
    for i, file in enumerate(csv_files):
        df = pd.read_csv(file, header=None)
        file_name = os.path.splitext(os.path.basename(file))[0]

    #calculate mean for denominator
        #sum elements
        means[i] += df.sum().sum(axis=0) 
        
        #count non-zero elements to calculate mean
        bool_df = df.astype(bool)
        non_zero_count = bool_df.sum().sum()
        
        #calculate mean
        means[i] /= non_zero_count
        
        #normalize data
        normalized_grid = df / means[i]

        normalized_path = os.path.join(currdir, f'{file_name}_norm.csv')
        normalized_grid.to_csv(normalized_path, index = False, header = False)

def Average_CSVs(args, currdir, state_files):
    print("Entered Average_CSVs")
    if args.norm:
        flagType = 'velocity_norm.csv'
    else:
        flagType = 'velocity.csv'

    all_files = glob.glob(f'{currdir}\\*{flagType}')

    #initialize sum dfs for means by state
    state_list = list(state_files.keys())
    state_df_dict = {}
    for state in state_list:
        df = pd.DataFrame(np.zeros((grid_size, grid_size)))
        state_df_dict[f'{state}'] = df

    #add all csvs to respective sum dfs
    for file in all_files:
        df = pd.read_csv(file, header=None)
        file_name = os.path.splitext(os.path.basename(file))[0].replace('_velocity_norm','').replace('_velocity','')
        state_df_dict[state_identifier(file_name, state_list)] += df
    
    #average by number of files
    for state in state_df_dict.keys():
        state_df_dict[state] /= len(state_files[state])
        state_df_dict[state].to_csv(os.path.join(currdir, f'{state}_velocity_average.csv'), index = False, header = False)
   
def Num_Waves_Comp(currdir, state_files):
    print("Entered Num_Waves_Comp")
    dict = {}
    flagType = '_numwaves'
    csv_files = glob.glob(f'{currdir}\\*{flagType}.csv')
    state_list = list(state_files.keys())

    #iterate through all csvs, read data and store in dictionary by state
    for file in csv_files:
        file_name = os.path.splitext(os.path.basename(file))[0]
        file_name_no_flag = file_name.replace(flagType,'')
        file_name_no_state = file_name_no_flag.replace(state_identifier(file_name_no_flag, state_list),'')

        with open(os.path.join(currdir, f'{file_name}.csv'), 'r') as f:
            csv_reader = csv.reader(f)
            state_data = next(csv_reader)
        
        if file_name_no_state not in dict:
            dict[file_name_no_state] = [0] * len(state_list)
        dict[file_name_no_state][state_list.index(state_identifier(file_name_no_flag, state_list))] = int(state_data[0])
    
    df = pd.DataFrame.from_dict(dict, orient='index', columns=state_list)
    df.to_csv(os.path.join(currdir, 'numwaves_comp.csv'), index = True, mode='w+')

    #individual bar graphs
    for mouse in dict.keys():
        x = df.loc[mouse]
        plt.bar(state_list,x)
        #labels
        plt.ylabel('Number of Waves')
        plt.xlabel('State')
        plt.title(f'{mouse} number of waves')
        #appearance
        plt.tight_layout()
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
       
        plt.savefig(os.path.join(currdir, f'{mouse}_numwaves.png'))
        plt.close()

    plt.clf()

    #line plot
    for mouse in dict.keys():
        x = df.loc[mouse]
        x_masked = np.where(x == 0, np.nan, x)
        plt.plot(state_list, x_masked, label=mouse, marker='o', linewidth=2)
    #labels
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')
    plt.title('Number of waves across states')
    plt.ylabel('Number of Waves')
    plt.xlabel('State')
    #appearance
    plt.tight_layout()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.savefig(os.path.join(currdir, 'numwaves_comp.png'))
    plt.close()
    plt.clf()

    #total waves per state bar graph
    df_total = df.sum(axis = 0) #sum up rows of table
    x = df_total
    plt.bar(state_list, x)
    #labels
    plt.title('Total number of waves')
    plt.ylabel('Number of Waves')
    plt.xlabel('State')
    #appearance
    plt.tight_layout()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.savefig(os.path.join(currdir, 'numwaves_total.png'))
    plt.close()
    plt.clf()

def Avg_Polar(data_path, args, data, currdir):
    all_files = [entry['filename'] for entry in data]

    print(f"Entered Avg_Polar for {all_files}")

    # Master list containing every x and y coord based on parameter/argument
    all_directionY = []
    all_directionX = []
    # Lists for normalized averages after taking individual averages
    avg_x_normalized = []
    avg_y_normalized = []

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
    ax.set_title('All waves have equal weight')

    ax1.hist(angles_norm, bins=36, range=(-np.pi, np.pi), density=True)
    ax1.set_title('Each wave weighted by # of vectors')
    
    # Plot the weighted average line
    ax.plot([0, weighted_average_angle], [0, ax.get_ylim()[1]], color='red', linewidth=2)
    ax1.plot([0, weighted_average_angle1], [0, ax1.get_ylim()[1]], color='black', linewidth=2)

    setOutputName = "avg_polar"
    fig.suptitle(f"Average for {args.n} {'mouse' if args.n == '1' else 'mice'}", fontsize=16)

    fig_path = os.path.join(currdir, f'{setOutputName}.png')
    fig.savefig(fig_path)
    plt.close()

def Planarity_Comp(currdir, state_files):
    print("Entered Planarity_Comp")
    dict = {}
    flagType = '_planarity'
    csv_files = glob.glob(f'{currdir}\\*{flagType}.csv')

    state_list = list(state_files.keys())

    for file in csv_files:
        file_name = os.path.splitext(os.path.basename(file))[0]
        file_name_no_flag = file_name.replace(flagType,'')
        file_name_no_state = file_name_no_flag.replace(state_identifier(file_name_no_flag, state_list),'')

        with open(os.path.join(currdir, f'{file_name}.csv'), 'r') as f:
            csv_reader = csv.reader(f)
            state_data = next(csv_reader)
        
        if file_name_no_state not in dict:
            dict[file_name_no_state] = [0] * len(state_list)
        dict[file_name_no_state][state_list.index(state_identifier(file_name_no_flag, state_list))] = float(state_data[0])
    
    df = pd.DataFrame.from_dict(dict, orient='index', columns=state_list)
    df.to_csv(os.path.join(currdir, 'planarity_comp.csv'), index = True, mode='w+')

    #individual bar graphs
    for mouse in dict.keys():
        x = df.loc[mouse]
        plt.bar(state_list,x)
        plt.ylabel('Planarity')
        plt.xlabel('State')
        plt.title(f"{mouse} planarity")
        plt.savefig(os.path.join(currdir, f'{mouse}_planarity.png'))
        plt.close()
    plt.clf()
    #line plot
    for mouse in dict.keys():
        x = df.loc[mouse]
        x_masked = np.where(x == 0, np.nan, x)
        plt.plot(state_list, x_masked, label=mouse, marker='o', linewidth=2)
    #labels
    plt.ylabel('Planarity')
    plt.xlabel('State')
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')
    plt.title('Planarity across states comparison')
    #appearance
    plt.tight_layout()

    plt.savefig(os.path.join(currdir, 'planarity_comp.png'))
    plt.close()
    plt.clf()

def Velocity_Topo_Average(args, currdir):
    #group csvs depending if norm flag is called
    if args.norm:
        print("Entered Velocity_Topo_Average with Norm Flag")
        csv_files = glob.glob(f'{currdir}\\*velocity_norm.csv')

    else:
        print("Entered Velocity_Topo_Average")
        csv_files = glob.glob(f'{currdir}\\*velocity.csv')
    
    #create individual heatmaps
    for file in csv_files:
        df = pd.read_csv(file, header=None)
        file_name = os.path.splitext(os.path.basename(file))[0]
        title_name = file_name = os.path.splitext(os.path.basename(file))[0].replace('_velocity_norm','').replace('_velocity','')

        #initialize empty 128x128 grid
        row_shape, col_shape = df.shape
        heat_data = np.zeros((row_shape, col_shape))
        
        #retrieve data from csv
        for row_index, row in df.iterrows():
            for col_index, col in enumerate(row):
                heat_data[row_index, col_index] = float(row[col_index])
        
        #topo parameters
        plt.figure(figsize=(10, 8))
        heatmap = sns.heatmap(heat_data, 
                              annot=False, 
                              cmap= "viridis", 
                              robust=True, 
                              mask=(heat_data == 0), 
                              xticklabels=False, 
                              yticklabels=False, 
                              square=True, 
                              cbar=True)
        heatmap.tick_params(left=False, bottom=False)
        heatmap.set_aspect('equal')
        heatmap.invert_yaxis()
        heatmap.set_title(f'{title_name} velocity heatmap', fontsize=18, loc="center")
        plt.savefig(os.path.join(currdir, f'{file_name}_topo.png'))
        plt.close()

    #group average csvs
    average_csv_files = glob.glob(f'{currdir}\\*average.csv')

    #create average heatmaps
    for file in average_csv_files:
        file_name = os.path.splitext(os.path.basename(file))[0]
        title_name = file_name = os.path.splitext(os.path.basename(file))[0].replace('velocity_norm','').replace('velocity','')
        
        df = pd.read_csv(file, header=None)
        row_shape, col_shape = df.shape
        average_heat_data = np.zeros((row_shape, col_shape))
       
        #retrieve data from csv
        for row_index, row in df.iterrows():
            for col_index, col in enumerate(row):
                average_heat_data[row_index, col_index] = float(row[col_index])

        #topo parameters
        plt.figure(figsize=(10, 8))
        heatmap = sns.heatmap(heat_data, 
                                annot=False, 
                                cmap= "viridis", 
                                robust=True, 
                                mask=(average_heat_data == 0),
                                xticklabels=False, 
                                yticklabels=False, 
                                square=True, 
                                cbar=True,
                                vmax=3.0) #max value
        heatmap.tick_params(left=False, bottom=False)
        heatmap.set_aspect('equal')
        heatmap.invert_yaxis()
        heatmap.set_title(f'{title_name} velocity heatmap', fontsize=18, loc="center")

        plt.savefig(os.path.join(currdir, f'{file_name}_topo.png'))
        plt.close()

def Num_Waves_Topo_Cumulative(currdir, state_files):
    print("Entered Num_Waves_Topo_Cumulative")
    dict = {}
    flagType = '_numwaves_topo'
    csv_files = glob.glob(f'{currdir}\\*{flagType}.csv') 
    state_list = list(state_files.keys())
    
    #initialize empty 128x128 grid for each state
    for state in state_list:
        dict[state] = np.zeros((grid_size, grid_size))
    
    #initialize count for each state
    count = [[[0 for _ in range(grid_size)] for _ in range(grid_size)] for _ in range(len(state_list))]
    
    #iterate through all csvs and add to respective state grid
    for file in csv_files:
        channel_list = np.genfromtxt(file, delimiter=',')

        file_name = os.path.splitext(os.path.basename(file))[0]
        file_name_no_flag = file_name.replace(flagType,'')

        for col, channel in enumerate(channel_list):
            for row, value in enumerate(channel):
                dict[state_identifier(file_name_no_flag, state_list)][col][row] += channel_list[col][row]
                count[state_list.index(state_identifier(file_name_no_flag, state_list))][col][row] += 1
            
    for state in state_list:
        #average by state
        dict[state] = np.divide(dict[state], count[state_list.index(state)])

        grid = dict[state]

        #check if there are any non-zero values
        if np.any(~np.isnan(grid)):
            #save csv
            np.savetxt(f"{currdir}\\{state}_numwaves_topo.csv", grid, delimiter=',') 
            #figure parameters
            plt.figure(figsize=(10, 8)) 
            heatmap = sns.heatmap(grid, 
                                    annot=False, 
                                    cmap= "viridis", 
                                    robust=True, 
                                    xticklabels=False, 
                                    yticklabels=False, 
                                    square=True, 
                                    cbar=True)
            heatmap.tick_params(left=False, bottom=False)
            heatmap.set_aspect('equal')
            heatmap.invert_yaxis()
            heatmap.set_title(f"{state} state number of waves", fontsize=18, loc="center")

            plt.savefig(os.path.join(currdir, f'{state}_numwaves_topo.png'))
            plt.close()
    
def Freq_Waves_Comp(currdir, state_files):
    print("Entered Freq_Waves_Comp")
    dict = {}
    flagType = '_freqwaves'
    csv_files = glob.glob(f'{currdir}\\*{flagType}.csv')
    state_list = list(state_files.keys())

    #iterate through all csvs, read data and store in dictionary by state
    for file in csv_files:
        file_name = os.path.splitext(os.path.basename(file))[0]
        file_name_no_flag = file_name.replace(flagType,'')
        file_name_no_state = file_name_no_flag.replace(state_identifier(file_name_no_flag, state_list),'')
        
        with open(os.path.join(currdir, f'{file_name}.csv'), 'r') as f:
                csv_reader = csv.reader(f)
                state_data = next(csv_reader)
            
        if file_name_no_state not in dict:
            dict[file_name_no_state] = [0] * len(state_list)
        dict[file_name_no_state][state_list.index(state_identifier(file_name_no_flag, state_list))] = float(state_data[0])
        
    #save to csv
    df = pd.DataFrame.from_dict(dict, orient='index', columns=state_list)
    df.to_csv(os.path.join(currdir, 'freqwaves_comp.csv'), index = True, mode='w+')
    
    #individual bar graphs
    for mouse in dict.keys():
        x = df.loc[mouse]
        plt.bar(state_list,x)
        #labels
        plt.ylabel('Wave Frequency')
        plt.xlabel('State')
        plt.title(f'{mouse} wave frequency')
        #appearance
        plt.tight_layout()
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
       
        plt.savefig(os.path.join(currdir, f'{mouse}_freqwaves.png'))
        plt.close()
    plt.clf()
    
    #line plot
    for mouse in dict.keys():
        x = df.loc[mouse]
        x_masked = np.where(x == 0, np.nan, x)
        plt.plot(state_list, x_masked, label=mouse, marker='o', linewidth=2)
    #labels
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')
    plt.title('Wave frequency across states')
    plt.ylabel('Wave Frequency')
    plt.xlabel('State')
    #appearance
    plt.tight_layout()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.savefig(os.path.join(currdir, 'freqwaves_comp.png'))
    plt.close()
    plt.clf()

    #total waves per state bar graph
    df_total = df.sum(axis = 0) #sum up rows of table
    x = df_total
    plt.bar(state_list, x)
    #labels
    plt.title('Total wave frequency')
    plt.ylabel('Wave Frequency')
    plt.xlabel('State')
    #appearance
    plt.tight_layout()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.savefig(os.path.join(currdir, 'freqwaves_total.png'))
    plt.close()
    plt.clf()

def Freq_Waves_Topo_Cumulative(currdir, state_files):
    print("Entered Freq_Waves_Topo_Cumulative")

    dict = {}
    flagType = 'freqwaves_topo'
    csv_files = glob.glob(f'{currdir}\\*{flagType}.csv') 
    state_list = list(state_files.keys())
    
    #initialize empty 128x128 grid for each state
    for state in state_list:
        dict[state] = np.zeros((grid_size, grid_size))
        
    #initialize count for each state
    count = [[[0 for _ in range(grid_size)] for _ in range(grid_size)] for _ in range(len(state_list))]
    
    #iterate through all csvs and add to respective state grid
    for file in csv_files:
        channel_list = np.genfromtxt(file, delimiter=',')

        file_name = os.path.splitext(os.path.basename(file))[0]
        file_name_no_flag = file_name.replace(flagType,'')

        for col, channel in enumerate(channel_list):
            for row, value in enumerate(channel):
                dict[state_identifier(file_name_no_flag, state_list)][col][row] += channel_list[col][row]
                count[state_list.index(state_identifier(file_name_no_flag, state_list))][col][row] += 1

    for state in state_list:
        #average by state
        dict[state] = np.divide(dict[state], count[state_list.index(state)])

        grid = dict[state]

        #check if there are any non-zero values
        if np.any(~np.isnan(grid)):
            #save csv
            np.savetxt(f"{currdir}\\{state}_freqwaves_topo.csv", grid, delimiter=',') 
            #figure parameters
            plt.figure(figsize=(10, 8)) 
            heatmap = sns.heatmap(grid, 
                                    annot=False, 
                                    cmap= "viridis", 
                                    robust=True, 
                                    xticklabels=False, 
                                    yticklabels=False, 
                                    square=True, 
                                    cbar=True)
            heatmap.tick_params(left=False, bottom=False)
            heatmap.set_aspect('equal')
            heatmap.invert_yaxis()
            heatmap.set_title(f"{state} state wave frequency", fontsize=18, loc="center")

            plt.savefig(os.path.join(currdir, f'{state}_freqwaves_topo.png'))
            plt.close()

def run(data_path, args, data, currdir, state_files):
    #average polar histogram across all mice
    Avg_Polar(data_path, args, data, currdir)
    if args.norm:
        Normalize(currdir)
    if args.avg:
        Average_CSVs(args, currdir, state_files)
    if args.v:
        Velocity_Violin(args, currdir) 
    if args.p:
        Planarity_Comp(currdir, state_files)
    #number of waves per state per mouse
    Num_Waves_Comp(currdir, state_files) 
    Num_Waves_Topo_Cumulative(currdir, state_files)
    if args.freq:
        #number of waves per recording length per state per mice
        Freq_Waves_Comp(currdir, state_files)
        Freq_Waves_Topo_Cumulative(currdir, state_files)

    Velocity_Topo_Average(args, currdir)
    print("Run Complete")