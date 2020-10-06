import pandas as pd
houses = pd.read_csv("california_housing.csv")

for column in houses.columns:
    print(houses[column].max())