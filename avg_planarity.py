import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
from num_waves import similar, data, states, mice
import numpy as np

table = []
for i in range(7):
    table.append([])
    for j in range(3):
        table[i].append(0)
j=0
for i, file in enumerate(data):
    filename = file.strip()
    df = pd.read_csv(Path(f"D:\\{filename}\\stage05_wave_characterization\\label_planar\\wavefronts_label_planar.csv"), usecols=['planarity'])
    mean = df['planarity'].mean()
    if 'WAKE' in str(filename):
        table[j][0] = mean
    elif 'NREM' in str(filename):
        table[j][1] = mean
    elif 'REM' in str(filename):
        table[j][2] = mean
    if i + 1 < len(data): #if not similar, goes to next list in list
        if not similar(data[i], data[i+1]):
            j += 1

df = pd.DataFrame({'States': states})
mice_unique = []
[mice_unique.append(item) for item in mice if item not in mice_unique]
for index, column in enumerate(table):
    df.insert(index, mice_unique[index], table[index], True)
state = df.pop('States')
df.insert(0, state.name, state)
df.to_csv(Path("D:\\Sandro_Code\\planarity\\avg_planarity.csv"), index = False, mode='w+')
print(df)

#individual bar graphs
for mouse in mice_unique:
    y = ['WAKE', 'NREM', 'REM']
    x = df[mouse]
    bgraph = plt.bar(y,x)
    bgraph[0].set_color('red')
    bgraph[1].set_color('blue')
    bgraph[2].set_color('green')
    plt.ylabel('average planarity')
    plt.title(mouse)
    plt.savefig(f"D:\\Sandro_Code\\planarity\\{mouse}_avg_planarity.png")
plt.clf()

#line plot
for mouse in mice_unique:
    y = ['WAKE', 'NREM', 'REM']
    x = df[mouse]
    x_masked = np.where(x == 0, np.nan, x)
    plt.plot(y, x_masked, label=mouse, marker='o', linewidth=2)
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')
plt.title('Average Planarity Across States Comparison')
plt.tight_layout()
plt.savefig('D:\\Sandro_Code\\planarity\\avg_planarity_comparison.png')
plt.clf()