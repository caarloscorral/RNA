import pandas as pd
houses = pd.read_csv("california_housing.csv")

for column in houses.columns:
    max = houses[column].max()
    min = houses[column].min()
    minus = (max-min)
    for index, value in enumerate(houses[column]):
        if index < 3:
            houses[column][index] = (value-min)/minus
            print(houses[column][index])
    