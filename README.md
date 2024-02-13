# Landsness Imaging COBRAWAP Project
Graphing COBRAWAP data

# Description
Examining the Sleep Slow Wave changes propagating across the cortex. Visualize wave characteristics comparing/contrasing 4 behavioral states: Wake(W), NREM (NR), REM (R). This is a data visualization tool that takes processed data from the  [COBRAWAP](https://github.com/NeuralEnsemble/cobrawap) pipeline and returns measures to graph the dynamic wave-like activity patterens found in the data. 

# Instructions


# Setting up the Development Environment

## 1. Install Visual Studio Code (VSCode)

Visual Studio Code is a popular code editor that provides a great environment for Python development. You can download and install VSCode from the official website: [Visual Studio Code](https://code.visualstudio.com/).

After the installation, launch VSCode.

## 2. Python Installation

To use Python with VSCode, you need to have Python installed. Follow these steps to install Python:

- Download Python from the official website: [Python Downloads](https://www.python.org/downloads/).
- Choose the version of Python that you want to install (e.g., Python 3.8 or Python 3.9).
- During installation, make sure to check the option that says "Add Python to PATH." This allows you to run Python from the command line.

To verify that Python is installed correctly, open the command prompt or terminal and run the following command:

```bash
python --version
```

This should display the version of Python you installed (e.g., Python 3.8.10).

## 3. Install Visual Studio Code Extensions

VSCode supports extensions that enhance your development environment. For Python development, install the "Python" extension:

- Open VSCode.
- Go to the Extensions view by clicking on the Extensions icon in the Activity Bar on the side of the window (or use the keyboard shortcut `Ctrl+Shift+X`).
- Search for "Python" in the Extensions view.
- Click the "Install" button for the official "Python" extension by Microsoft.

This extension provides features like code linting, debugging, and code completion for Python.

## 4. Open Your Code

Open your Python code project in VSCode:

- Click on "File" in the menu.
- Select "Open Folder" and navigate to the directory where your code is located.
- Click "Open."

You can now start working with your code in VSCode.

# Prerequisites

The prerequisites for running the code include specific Python libraries, data files, and additional dependencies. Make sure you've fulfilled these prerequisites as mentioned in your code's manual. Install them using the `pip` command.

To make sure that the latest version of pip is updated/installed

```bash
python -m pip install -U pip
```

Now to install the dependencies run these set of commands 

```bash
python -m pip install pandas
```

```bash
python -m pip install matplotlib
```

```bash
python -m pip install pathlib
```

```bash
python -m pip install scipy
```

```bash
python -m pip install seaborn
```


# Running Your Code

- Open the integrated terminal in VSCode by clicking on "View" in the menu and selecting "Terminal."
- Navigate to the directory where your code is located using the `cd` command.
- Run your code using the appropriate Python command, providing the script name as an argument.

For example, to run `run_script.py`:

```bash
python run_script.py
```
Using Tkinter, you will be able to select the directory in which the data/files/inputs to analyze are located. An example of this window is provided below
<img width="943" alt="Screenshot 2024-01-10 at 3 53 55 AM" src="https://github.com/cilantroxiao/landsness_imaging/assets/79768734/89df1800-8696-4182-af45-cce4a2eebf93">

Once the directory is selected, an output folder specifying the date and time of the run will be created including all outputs as well as the .txt file with the parameters and inputs you selected.

# Inputs
## input.txt

The input text file contains a series of command lines. Each command line specifies the execution of Python scripts with certain arguments. The flags are as follows:

  * --norm: Specifies normalization of data.
  * --out 'directory'/'folder': Specifies the output directory or folder.
  * --avg: Specifies average after data has been processed.
  * --v: Specifies violin plot of velocities.
  * --n 'num': Specifies number of Mice 
  * --p: Specifies planarity
  * --num: Specifies number of waves.
  * --f: Specifies input
  
Here's an example below of how your input.txt should look:
Format should be "Filename:wave_ids" or "Directory:wave_ids". If you want all wave_ids analyzed then just include the "filename".

```python
#filename:wave_ids --out .\\output --norm
--f SLEEP_119_2_NREM_54 --norm
--f CUMULATIVE --avg --v --num --p --norm --n 1 #This line specifies flags to run Step2.py.
```

## run_script.py

This Python script reads commands from an input text file and executes them. The input file is specified using the 'input_path' variable. Each line in the input file corresponds to a command that is executed by invoking Python scripts named "step1.py." After processing all commands, in other words when the CUMLATIVE tag is reached (representing the final line) the script runs "step2.py" with the last command from the input file.

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

