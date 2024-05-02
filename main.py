#imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

#Reading the dataset
data = pd.read_csv('data/data.csv')
print(data.head())
print(data.info())

#cleaning the dataset
data.drop('Unnamed: 32', axis=1, inplace=True)
data.drop('id', axis=1, inplace=True)
data['diagnosis'] = data['diagnosis'].apply(lambda val:1 if val=='M' else 0)

#checcking for missing values
print(data.isnull().sum())

#Correlation HeatMap 
corrMatrix = data.corr()
mask = np.triu(np.ones_like(corrMatrix, dtype=bool))

plt.figure(figsize=(20,15))
sns.heatmap(corrMatrix, mask=mask, cmap='coolwarm', annot=True, linewidths=2, fmt = ".2f")
plt.title("Coorelation Heatmap")
plt.show()
