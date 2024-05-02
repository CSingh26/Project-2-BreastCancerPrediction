#imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

#Reading the dataset
data = pd.read_csv('data/data.csv')
print(data.head())
print(data.info())

#cleaning the dataset
data.drop('Unnamed: 32', axis=1, inplace=True)
print(data.info())

#checcking for missing values
print(data.isnull().sum())

