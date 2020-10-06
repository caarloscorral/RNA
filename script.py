import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
data = pd.read_csv("california_housing.csv")

for column in data.columns:
    max = data[column].max()
    min = data[column].min()
    minus = (max-min)
    data[column] -= min
    data[column] /= minus

training, testing = train_test_split(data, shuffle=True, test_size=0.4)
testing, validation = train_test_split(testing, shuffle=True, test_size=0.5)

print(training)
print(testing)
print(validation)
    