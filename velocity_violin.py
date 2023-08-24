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
from scipy.stats import gaussian_kde
import seaborn as sns
grid_size = 128
def velocity_violin():
    v_list = np.empty(grid_size**2)

    csv_files = glob.glob('D:\\Sandro_Code\\channel_wise_velocity\\*velocity.csv')
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
        
        if index + 1 < len(data):
            fig, ax = plt.subplots()
            kde = gaussian_kde(v_list)
            scaling_factor = 1 / np.max(kde(v_list))
            sns.violinplot(data=[v_list * scaling_factor], vert=False, ax=ax, bw='scott')

            fig.suptitle(f"{file_name} Velocity Density Distribution", fontsize=15)
            ax.set_ylabel("log10 channel velocity [mm/s]")
            print(v_list)
            plt.savefig(f"D:\\Sandro_Code\\velocity_violins\\{file_name}violin.png")
velocity_violin()