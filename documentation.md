# Documentation

## run_script.py

This Python script reads commands from an input text file and executes them. The input file is specified using the 'input_path' variable. Each line in the input file corresponds to a command that is executed by invoking Python scripts named "step1.py." After processing all commands, in other words when the CUMLATIVE tag is reached (representing the final line) the script runs "step2.py" with the last command from the input file.

## input.txt 
Each command line flag specifies the execution of different functions with certain arguments. Format should be "Filename:wave_ids" or "Directory:wave_ids". If you want all wave_ids analyzed then just include the "filename".

The flags are as follows:

* `--f 'filename'`: Specifies input filenames and associated wavefronts.
* `--norm`: Specifies normalization of data.
* `--avg`: Specifies averaging data at the end.
* `--v`: Specifies the creation of a violin plot of velocities.
* `--p`: Specifies planarity.
* `--num`: Specifies the number of waves.
* `--n 'number'`: Specifies the number of mice.
