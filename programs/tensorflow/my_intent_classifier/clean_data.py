import pandas as pd
import numpy as np

df = pd.read_csv('data/raw_car_dashboard.csv', sep=',')

# X_train = df['utterances']
# Y_train = df['intent']

dummy_variable_1 = pd.get_dummies(df["class"])

# print(dummy_variable_1.head())

df = pd.concat([df, dummy_variable_1], axis=1)
df.drop("class", axis = 1, inplace=True)

# print(df.head())

df.to_csv('data/data.csv', encoding='utf-8', index=False)
