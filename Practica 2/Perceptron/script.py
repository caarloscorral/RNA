# Importing libraries
import random
import os
import numpy as np
import pandas as pd
# from sklearn.model_selection import cross_validation
def ceildiv(a, b):
    return -(-a // b)
############################################################# PREPROCESSING #############################################################

# Getting data
data = pd.read_csv(r"Practica 2/Perceptron/datosNubes.txt", sep='\t')
maxx = {}
minn = {}

# Normalize data
for column in data.columns[:len(data.columns)-1]:
    maxx[column] = data[column].max()
    minn[column] = data[column].min()
    minus = (maxx[column] - minn[column])
    data[column] -= minn[column]
    data[column] /= minus

data = data.sample(frac=1)
data = data.to_numpy()
numberOfFolds = 4
folds = [[], [], [], []]

total = {}
total['cieloDespejado'] = 48
total['multinube'] = 156
total['nube'] = 513

count = {}
count['cieloDespejado'] = 0
count['multinube'] = 0
count['nube'] = 0

for row in data:
    out = row[-1]
    index = count[out]//(total[out]//numberOfFolds)
    count[out] += 1 if count[out] < (total[out] - 3) else 0
    fold = folds[index]
    fold.append(row)

for index, fold in enumerate(folds):
    count['cieloDespejado'] = 0
    count['multinube'] = 0
    count['nube'] = 0
    for row in fold:
        out = row[-1]
        count[out] += 1
    print('fold ' + str(index)+':', count)

dataframes = list(map(pd.DataFrame, folds))
for index,df in enumerate(dataframes):
    df.to_csv(r"Practica 2\Perceptron\Outputs\fold" + str(index) + ".csv", index=False)
