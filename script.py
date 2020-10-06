import pandas as pd
from sklearn.utils import shuffle
data = pd.read_csv("california_housing.csv")

for column in data.columns:
    max = data[column].max()
    min = data[column].min()
    minus = (max-min)
    data[column] -= min
    data[column] /= minus

data = data.sample(frac=1).reset_index(drop=True)
training = data.sample(frac=0.6).reset_index(drop=True)
validation = data.sample(frac=0.2).reset_index(drop=True)
test = data.sample(frac=0.2).reset_index(drop=True)

print(training)
print(validation)
print(test)
    