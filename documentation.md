# Tool Structure

## run_script.py

This Python script reads commands from an input text file and executes them. The input file is specified using the 'input_path' variable. Each line in the input file corresponds to a command that is executed by invoking Python scripts named "step1.py." After processing all commands, in other words when the CUMLATIVE tag is reached (representing the final line) the script runs "step2.py" with the last command from the input file.

## input.txt 
The flags are as follows:

  * --norm: Specifies normalization of data.
  * --out 'directory'/'folder': Specifies the output directory or folder.
  * --avg: Specifies average after data has been processed.
  * --v: Specifies violin plot of velocities.
  * --n 'num': Specifies number of Mice 
  * --p: Specifies planarity
  * --num: Specifies number of waves.
  * --f: Specifies input




# Non-Helper Functions:
## Step 1:
### `create_parser()`
This function sets up an ArgumentParser for command-line argument parsing. It defines and handles command-line arguments, such as input file, output directory, and normalization flag, to control the program's behavior.

### `parse_waves()`
This function extracts the filename and wave IDs from the user-provided input. It handles cases where the input is in the format of 'filename:wave_ids' and determines the range of wave IDs to consider.

### `Polar_Histogram(filename, wave_ids)`
Generates polar histograms based on the provided `filename` and wave IDs. It processes and normalizes data from CSV files, calculating the weighted average angle for each dataset and creating polar histograms for visualization.

### `Individual_CSVs(filename, wave_ids)`
Creates CSV files containing velocity data based on the specified `filename` and wave IDs. It accumulates and averages velocity data and saves the results in CSV format.

### `Heat_Mapper(norm, avg)`
Generates heatmaps based on the program's configuration. It can produce velocity heatmaps, normalized heatmaps, or average heatmaps, depending on the arguments provided.

## Step 2:
### `create_parser()`
This function sets up an ArgumentParser for command-line argument parsing. It defines and handles command-line arguments, such as input file, output directory, and normalization flag, to control the program's behavior.

### `Velocity_Violin()`
This function generates violin plots of velocity data extracted from CSV files. It processes multiple CSV files and visualizes the velocity distribution.

### `Normalize()`
This function normalizes CSV data, specifically related to velocity. It processes CSV files and normalizes the data within those files.

### `Average_CSVs()`
This function computes average values from CSV files, especially relevant for velocity data. It divides the data by the number of files and stores the resulting averages in new CSV files.

### `Avg_Planarity()`
This function calculates and graphs the average planarity across different states such as WAKE, NREM, and REM. It produces individual bar graphs, line plots, and a summary comparison graph.

### `Num_Waves()`
This function calculates and graphs the number of waves for each state, including WAKE, NREM, and REM. It produces individual bar graphs, a line plot, and a total bar graph for comparison.



# Helper Functions:
## Step 1:
### `add(list, row, column, df)`
A helper function that adds values from multiple DataFrames, specified by a list, into a target DataFrame (`df`) at a given row and column index.

### `divide(df, size)`
This function normalizes the values in a DataFrame (`df`) by dividing each element by a specified `size` value. It is used for normalizing the data.

### `similar(a, b)`
This method checks the similarity between two strings (`a` and `b`) by comparing and removing certain substrings, specifically the elements in the `states` list. It uses the `SequenceMatcher` to calculate the similarity ratio.
