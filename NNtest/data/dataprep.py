import sys, os
import random

with open('data.csv') as fin:
    lines = fin.readlines()

datatypes = []
reciptypes = []
condtypes = []
traindata = []
validdata = []
testdata = []
for line in lines[1:]:
    label, data, recip, cond = line.strip().split(',')
    if data not in datatypes:
        datatypes.append(data)
    if recip not in reciptypes:
        reciptypes.append(recip)
    if cond not in condtypes:
        condtypes.append(cond)
    randnumber = random.random()
    if randnumber < 0.8:
        traindata.append(line)
    elif randnumber < 0.9:
        validdata.append(line)
    else:
        testdata.append(line)

with open('reciptypes.txt', 'w') as fout:
    for recip in reciptypes:
        fout.write(recip + '\n')
with open('datatypes.txt', 'w') as fout:
    for data in datatypes:
        fout.write(data + '\n')
with open('conditions.txt', 'w') as fout:
    for cond in condtypes:
        fout.write(cond + '\n')

with open('train.csv', 'w') as fout:
    fout.writelines(traindata)
with open('valid.csv', 'w') as fout:
    fout.writelines(validdata)
with open('test.csv', 'w') as fout:
    fout.writelines(testdata)
