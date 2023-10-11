# Landsness Imaging COBRAWAP Project
Graphing COBRAWAP data

# Description
Examining the Sleep Slow Wave changes propagating across the cortex. Visualize wave characteristics comparing/contrasing 4 behavioral states: Wake(W), NREM (NR), REM (R)

# Inputs
input.txt

# Non-Helper Functions:


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

