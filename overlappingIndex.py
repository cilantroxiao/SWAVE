import tkinter as tk
from tkinter import filedialog
import pandas as pd
import os

def open_directory():
    # Open file explorer to select the directory
    root = tk.Tk()
    root.withdraw()
    selected_directory = filedialog.askdirectory(title="Select Mouse Directory")
    
    if selected_directory:
        # Navigate to the `stage05_wave_characterization/time_stamp` folder
        stage05_path = os.path.join(selected_directory, 'stage05_wave_characterization', 'time_stamp')
        csv_file_path = os.path.join(stage05_path, 'wavefronts_time_stamp.csv')
        
        if os.path.exists(csv_file_path):
            # Read the CSV file
            df = pd.read_csv(csv_file_path)
            print("CSV file loaded successfully.")
            
            # Logic to group consecutive waves with timestamps within 0.5 seconds
            wave_groups = []
            current_group = [(df.loc[0, 'wavefronts_id'], df.loc[0, 'time_stamp'])]  # Start the first group
            
            for i in range(1, len(df)):
                current_id = df.loc[i, 'wavefronts_id']
                current_time = df.loc[i, 'time_stamp']
                previous_time = df.loc[i - 1, 'time_stamp']
                
                # If the difference between the current and previous timestamp is <= 0.5, add to current group
                if abs(current_time - previous_time) <= 0.5:
                    current_group.append((current_id, current_time))
                else:
                    # If not within 0.5 seconds, close the current group and start a new one
                    wave_groups.append(current_group)
                    current_group = [(current_id, current_time)]
            
            # Append the last group
            wave_groups.append(current_group)
            
            # Display the grouped waves with IDs and timestamps
            if wave_groups:
                print("Grouped wavefronts that may form a single wave (within 0.5 seconds):")
                for group in wave_groups:
                    if len(group) > 1:  # Only display groups with more than one wavefront
                        formatted_group = [f"(ID: {wave_id}, Time: {timestamp}s)" for wave_id, timestamp in group]
                        print(f"Wave group: {formatted_group}")
            else:
                print("No grouped wavefronts found.")
        else:
            print(f"CSV file not found at {os.path.abspath(csv_file_path)}.")
    else:
        print("No directory selected.")

if __name__ == "__main__":
    open_directory()
