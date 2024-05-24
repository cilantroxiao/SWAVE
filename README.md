# SWAVE
Sleep Wave Analysis Visualization Engine

## Description
Examining the Sleep Slow Wave changes propagating across the cortex. Visualize wave characteristics comparing/contrasing 3 behavioral states: Wake(W), NREM (NR), REM (R). This is a data visualization tool that takes processed data from the  [COBRAWAP](https://github.com/NeuralEnsemble/cobrawap) pipeline and returns measures to graph the dynamic wave-like activity patterens found in the data. 

## Documentation
[Documentation](https://github.com/cilantroxiao/landsness_imaging/blob/aly_code/documentation.md)


## Context/Justification/Content
![image](https://github.com/cilantroxiao/landsness_imaging/assets/79768734/ac2bddd8-be7e-4c60-9b1d-e136ead08ff6)

For researchers studying cortical wave-like activity and Peak/Trough state dynamics, effectively visualizing and interpreting the complex analysis results is crucial for gaining meaningful insights. While COBRAWAP provides a powerful and modular pipeline for analyzing wide-field calcium imaging data, seamless integration with intuitive visualization tools can further enhance the accessibility and interpretability of the analysis outcomes.

Our visualization tool is designed to complement and collaborate with COBRAWAP, forming a unified and user-friendly platform for comprehensive cortical wave analysis. By seamlessly integrating with COBRAWAP's outputs, our tool bridges the gap between raw analysis results and their effective interpretation, empowering researchers with both programming and non-programming backgrounds to explore and understand the intricate dynamics of cortical wave activity and Peak/Trough state detections.

Key features of our collaborative visualization solution include:

1. Polar Histograms: Gain insights into the directional properties of cortical waves by visualizing their propagation directions, facilitating the identification of dominant propagation patterns.
  <img width="500" alt="Polar Histogram Example" src="https://github.com/cilantroxiao/landsness_imaging/assets/79768734/bdd9c214-74f8-4396-bd93-9af27e0fec8f">

2. Velocity Heatmaps: Explore the spatial distribution of wave velocities across the imaging field, revealing regions with distinct wave propagation speeds and potential correlations with underlying cortical structures.
    <img width="500" alt="Velocity Heatmap Example" src="https://github.com/cilantroxiao/landsness_imaging/assets/79768734/972b635a-b196-43ea-8e36-2a8739e6bcdf">

3. Planarity Visualizations: Assess the planarity (cortical vs. subcortical) of the waves through intuitive visualizations that highlight deviations from planarity.

   <img width="500" alt="Planarity Example" src="https://github.com/cilantroxiao/landsness_imaging/assets/79768734/c5f9964e-d180-47c4-836e-f0016a27514f">

4. Wave Frequency: Analyze the number of waves per unit of time distribution of cortical waves, revealing patterns of activity across different regions.

  <img width="500" alt="Wave Incidents" src="https://github.com/cilantroxiao/landsness_imaging/assets/79768734/eda3feef-9b72-4993-b0e5-f37fbc2e1e2c">



5. FOR THE FUTURE: Locality Index to access local vs global waves


Together, COBRAWAP and our visualization tool form a comprehensive and accessible platform for cortical wave research, fostering interdisciplinary collaboration, knowledge sharing, and ultimately driving scientific progress in this field.

## Installation/Set-up
Before using our visualization tool, ensure that you have set up and run the COBRAWAP pipeline to process your data. Follow the installation and usage instructions provided in the [COBRAWAP](https://cobrawap.readthedocs.io/en/latest/pipeline.html) documentation to analyze your data and generate the necessary output files and file structure.

To run our tool, you'll need the following packages. Refer to these detailed instructions to [install](https://github.com/cilantroxiao/landsness_imaging/blob/aly_code/installation-setup.md#prerequisites) them: 

- pandas, matplotlib, numpy, pathlib, glob, csv, os, argparse, difflib, scipy, seaborn

## How to run your code

### input.txt

The input text file serves as a set of instructions for the visualization. Please refer to the [documentation](https://github.com/cilantroxiao/landsness_imaging/blob/aly_code/documentation.md#inputtxt) for more information.
  
An example of how input.txt could look:

```python
#filename:wave_ids --out .\\output --norm
--f 'insert_Mice_Name' --norm
--f 'insert_Mice_Name':1-30, 52, 54 --norm
--f CUMULATIVE --avg --v --num --p --norm --n 1 #This line specifies flags to run Step2.py.
```

### Running your Code
1. Open your terminal or command prompt.
2. Navigate to the directory where your code is located using the `cd` command.
3. Run the script/tool using the appropriate command, providing the necessary arguments if any.

For example, to run `run_script.py`:

```bash
python run_script.py
```
Using Tkinter, you will be able to select the directory in which the data/files/inputs to analyze are located. An example of this window is provided below
<img width="946" alt="Screenshot 2024-04-16 at 2 42 48 PM" src="https://github.com/cilantroxiao/landsness_imaging/assets/79768734/08779985-e5f3-4212-9b30-f53e31c211fe">

## Outputs
For each iteration, a timestamped folder will be created including a text file with parameters and the corresponding visuals
<img width="657" alt="Screenshot 2024-04-16 at 3 00 53 PM" src="https://github.com/cilantroxiao/landsness_imaging/assets/79768734/81ce3387-298a-4ed5-88c0-c837730f3b70">


## Liscenses / Legal


