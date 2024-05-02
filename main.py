#imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression

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
sns.heatmap(corrMatrix, mask=mask, cmap=sns.diverging_palette(220, 10, as_cmap=True), 
            annot=True, linewidths=2, fmt = ".2f", square=True)
plt.title("Coorelation Heatmap")
plt.show()

#Feature Selection

##Univariate Feature Selection
X = data.drop(['diagnosis'], axis=1)
y = data['diagnosis']

selector = SelectKBest(f_regression, k='all')
fit = selector.fit(X,y)

pVals = pd.DataFrame(fit.pvalues_)
scores = pd.DataFrame(fit.scores_)
inputVars = pd.DataFrame(X.columns)
stats = pd.concat([inputVars, pVals, scores], axis=1)
stats.columns = ("InputVariable", "p_values", "f_scores")
stats.sort_values(by='p_values', inplace=True)

pValueThreshold = 0.05
scoreThreshold = 5

varsSelected = stats.loc[(stats['f_scores'] >= scoreThreshold) & 
                         (stats['p_values'] <= pValueThreshold)]

varsSelected = varsSelected['InputVariable'].tolist()
XNew = X[varsSelected]

print(XNew.info())
