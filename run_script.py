import os
file = open('.\\input.txt', 'r')
tail = file.readlines()[-1]
file.seek(0)
for line in file:
    os.system(f'python step1.py {line}')
    if line == tail:
        break
os.system(f'python step2.py {tail}')