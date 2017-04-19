import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df_train = pd.read_csv('aggregated_2015.csv')
train, test = train_test_split(df_train, train_size=0.8)
test.to_csv('subset_aggregated_2015.csv', index=False, header=True)

df_test = pd.read_csv('aggregated_2016.csv')
train, test = train_test_split(df_train, train_size=0.8)
test.to_csv('subset_aggregated_2016.csv', index=False, header=True)