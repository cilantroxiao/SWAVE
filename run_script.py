import argparse
import os
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import pandas as pd
import step1
import step2


def create_parser():
    parser = argparse.ArgumentParser(
        prog='Title2',
        description='Description2',
        epilog='Hope this works2')
    parser.add_argument('--norm', action='store_true', help='normalize data?')
    parser.add_argument('--avg', action='store_true', help='avg data at end?')
    parser.add_argument('--v', action='store_true', help='violin plot of velocities?')
    parser.add_argument('--out', required=True, help='step1outputdir = step2inputdir and step2outputdir')
    parser.add_argument('--p', action='store_true', help='planarity')
    parser.add_argument('--num', action='store_true', help='num of waves')
    parser.add_argument('--polar', action='store_true', help='polar plot')
    parser.add_argument('--n', help='number of mice')
    parser.add_argument('filename_wave_ids', help='input filenames and associated wavefronts')
    return parser


def parse_waves(args):
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
        df = pd.read_csv(Path(f"{data_path}{filename}\\stage05_channel-wave_characterization\\direction_local\\wavefronts_direction_local.csv"))
        last_wave_id = df['wavefronts_id'].max()
        wave_ids.extend(range(1, last_wave_id)) # Set wave_ids_input as a range from 1 to last_wave_id 
    return filename, wave_ids


def run_command(args):
    filename, wave_ids = parse_waves(args)
    step1.run(data_path, filename, wave_ids, args)
    #step2.run(data_path, filename, wave_ids, args)


root = tk.Tk()
root.withdraw()
data_path = filedialog.askdirectory()

input_path = '.\\input.txt'

parser = create_parser()
with open(input_path, 'r') as f:
    for line in f:
        args = parser.parse_args(line.split())
        run_command(args)