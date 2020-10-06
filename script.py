import pandas as pd
data = pd.read_csv("california_housing.csv")

for column in data.columns:
    max = data[column].max()
    min = data[column].min()
    minus = (max-min)
    data[column] -= min
    data[column] /= minus

data = data.sample(frac=1).reset_index(drop=True)

print(data)
    