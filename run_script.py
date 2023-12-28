import argparse
import os
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import pandas as pd
import step1
import step2
import os
import time

def create_parser():
    parser = argparse.ArgumentParser(
        prog='Title2',
        description='Description2',
        epilog='Hope this works2')
    parser.add_argument('--f', help='input filenames and associated wavefronts')
    parser.add_argument('--norm', action='store_true', help='normalize data?')
    parser.add_argument('--avg', action='store_true', help='avg data at end?')
    parser.add_argument('--v', action='store_true', help='violin plot of velocities?')
    parser.add_argument('--p', action='store_true', help='planarity')
    parser.add_argument('--num', action='store_true', help='num of waves')
    parser.add_argument('--n', help='number of mice')
    return parser

def parse_wave_ids(wave_ids_input):
    wave_ids = []
    if wave_ids_input:
        for part in wave_ids_input.split(','):
            if '-' in part:
                start, end = part.split('-')[0], part.split('-')[1]
                wave_ids.extend(range(int(start), int(end) + 1))
            else:
                wave_ids.append(int(part))
    return wave_ids

def parse_waves(args):
    wave_ids = []
    if ':' in args.f: #if waves/ranges specified
        filename, wave_ids_input = args.f.split(':')
        wave_ids = parse_wave_ids(wave_ids_input)
    else: #all waves
        filename = args.f
        df = pd.read_csv(Path(f"{data_path}{filename}\\stage05_channel-wave_characterization\\direction_local\\wavefronts_direction_local.csv"))
        last_wave_id = df['wavefronts_id'].max()
        wave_ids_input = f'0-{last_wave_id}'
        wave_ids = parse_wave_ids(wave_ids_input)
    return filename, wave_ids

def create_new_directory(out_path):
    # Generate a unique directory name using the current timestamp
    timestamp = time.strftime("%d-%m-%Y-%H-%M-%S")
    new_directory_name = f"output_{timestamp}"

    # Create a new directory
    os.makedirs(f'{out_path}\\{new_directory_name}')

    print(f"New directory created: {new_directory_name}")
    return new_directory_name, timestamp

root = tk.Tk()
root.withdraw()
data_path = filedialog.askdirectory()

input_path = '.\\input.txt' #modify this
output_path = '.\\output' #modify this
data = []

parser = create_parser()
outdir, timestamp = create_new_directory(output_path)
currdir = f'{output_path}\\{outdir}'
with open(input_path, 'r') as f:
    contents = f.read()
    with open(f'{currdir}\\run_params_{timestamp}.txt', 'w') as f_copy:
        f_copy.write(contents)
        f_copy.close()
    f.seek(0)
    tail =  f.readlines()[-1].strip()
    f.seek(0)
    for line in f:
        args = parser.parse_args(line.split())
        if "CUMULATIVE" in line.strip():
            step2.run(data_path, args, data, currdir)
            break
        filename, wave_ids = parse_waves(args)
        data.append({'filename': filename, 'wave_ids': wave_ids})
        step1.run(data_path, filename, wave_ids, args, currdir)