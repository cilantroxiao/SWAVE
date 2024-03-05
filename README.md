# Landsness Imaging COBRAWAP Project
Graphing COBRAWAP data

# Description
Examining the Sleep Slow Wave changes propagating across the cortex. Visualize wave characteristics comparing/contrasing 4 behavioral states: Wake(W), NREM (NR), REM (R). This is a data visualization tool that takes processed data from the  [COBRAWAP](https://github.com/NeuralEnsemble/cobrawap) pipeline and returns measures to graph the dynamic wave-like activity patterens found in the data. 

# Links for Documentation

# Context/Justification/Content





# How to run your code

## run_script.py

This Python script reads commands from an input text file and executes them. The input file is specified using the 'input_path' variable. Each line in the input file corresponds to a command that is executed by invoking Python scripts named "step1.py." After processing all commands, in other words when the CUMLATIVE tag is reached (representing the final line) the script runs "step2.py" with the last command from the input file.

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
--f SLEEP_119_2_NREM_54:1-30, 52, 54 --norm
--f CUMULATIVE --avg --v --num --p --norm --n 1 #This line specifies flags to run Step2.py.
```

## Running your Code
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


# Outputs




# Liscenses / Legal


