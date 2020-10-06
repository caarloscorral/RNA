import pandas as pd
from sklearn.utils import shuffle
data = pd.read_csv("california_housing.csv")

for column in data.columns:
    max = data[column].max()
    min = data[column].min()
    minus = (max-min)
    data[column] -= min
    data[column] /= minus

data = shuffle(data)

    