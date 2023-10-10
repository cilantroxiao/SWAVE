import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.colors as mcolors
from difflib import SequenceMatcher
import glob
import csv
from step1 import states, similar, divide, add, Heat_Mapper
import os
import argparse
from scipy.stats import gaussian_kde
import seaborn as sns
grid_size = 128
def create_parser():
    parser = argparse.ArgumentParser(
        prog='Title2',
        description='Description2',
        epilog='Hope this works2')
    parser.add_argument('--norm', action='store_true', help='normalize data?')
    parser.add_argument('--avg', action='store_true', help='avg data at end?')
    parser.add_argument('--v', action='store_true', help='violin plot of velocities?')
    parser.add_argument('--out', required=True, help='step1outputdir = step2inputdir and step2outputdir')
    parser.add_argument('--n', help='number of mice')
    return parser
def Velocity_Violin():

    v_list = np.empty(grid_size**2)

    csv_files = glob.glob('{args.out}\\*velocity.csv')
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
        
        if index + 1 < args.n:
            fig, ax = plt.subplots()
            kde = gaussian_kde(v_list)
            scaling_factor = 1 / np.max(kde(v_list))
            sns.violinplot(data=[v_list * scaling_factor], vert=False, ax=ax, bw='scott')

            fig.suptitle(f"{file_name} Velocity Density Distribution", fontsize=15)
            ax.set_ylabel("log10 channel velocity [mm/s]")
            print(v_list)
            plt.savefig(f"D:\\Sandro_Code\\velocity_violins\\{file_name}_violin.png")
def Normalize():
    csv_files = glob.glob(f'{args.out}\\*velocity_N.csv')
    j = 0
    sums = [0] * len(args.n)
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
def Average_CSVs():
    grid_average = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
    averages_nrem, averages_rem, averages_wake = pd.DataFrame(grid_average), pd.DataFrame(grid_average), pd.DataFrame(grid_average)
    norm_path = f'{args.out}\\*velocity_N.csv'
    avg_path = f'{args.out}\\*velocity.csv'
    #args.norm
    if args.norm:
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
def Avg_Planarity():
    table = []
    for i in range(7):
        table.append([])
        for j in range(3):
            table[i].append(0)
    j=0
    csv_files = glob.glob('{args.out}\\*velocity.csv')
    for i, file in enumerate(csv_files):
        df = pd.read_csv(file, header=None)
        file_name = os.path.splitext(os.path.basename(file))[0].replace('velocity','')
        i = 0
        df = pd.read_csv(Path(f"D:\\{file_name}\\stage05_wave_characterization\\label_planar\\wavefronts_label_planar.csv"), usecols=['planarity'])
        mean = df['planarity'].mean()
        if 'WAKE' in str(file_name):
            table[j][0] = mean
        elif 'NREM' in str(file_name):
            table[j][1] = mean
        elif 'REM' in str(file_name):
            table[j][2] = mean
        if i + 1 < len(csv_files) and not similar(file_name, os.path.splitext(os.path.basename(csv_files[i+1]))[0]):
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
#need to fix
def Num_Waves():
    #0 -> Wake, 1 -> NREM, 2 -> REM
    table = []
    for i in range(7):
        table.append([])
        for j in range(3):
            table[i].append(0)
    #0 -> L1, 1 -> L3, 2 -> E2, 3 -> 119_2, 4 -> 328A_3, 5 -> 132_1, 6 -> 132_2

    #(because sometimes data is missing) compares names, if 1 then they are the same mouse. If not 1, the new mouse.
    j = 0
    for i, file in enumerate(data):
        filename = file.strip()
        df = pd.read_csv(Path(f"D:\\{filename}\\stage05_channel-wave_characterization\\velocity_local\\wavefronts_velocity_local.csv"), usecols=['wavefronts_id'])
        index = df.tail(1).index.item() #grabs last index's value in file
        num = df.at[index, 'wavefronts_id']
        if 'WAKE' in str(filename):
            table[j][0] = num
        elif 'NREM' in str(filename):
            table[j][1] = num
        elif 'REM' in str(filename):
            table[j][2] = num
        #print(f"{filename} {num}")
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
    df.to_csv(Path("D:\\Sandro_Code\\number_of_waves\\number_of_waves.csv"), index = False, mode='w+')
    #print(df)

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
        plt.savefig(f"D:\\Sandro_Code\\number_of_waves\\{mouse}_number_of_waves.pdf")
    plt.clf()

    #line plot
    for mouse in mice_unique:
        y = ['WAKE', 'NREM', 'REM']
        x = df[mouse]
        plt.plot(y, x, label=mouse, marker='o', linewidth=2)
    plt.legend()
    plt.title('Waves Across States Comparison')
    plt.savefig('D:\\Sandro_Code\\number_of_waves\\number_of_waves_comparison.pdf')
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
    plt.savefig('D:\\Sandro_Code\\number_of_waves\\total_number_of_waves.pdf')
    plt.clf()
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    if args.v:
        Velocity_Violin()
    if args.norm:
        Normalize()
    if args.avg:
        Average_CSVs()
    Heat_Mapper(args.norm, args.avg)
