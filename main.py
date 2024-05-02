#imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

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
#Helps to sort out the correlated features for a better fitted model
corrMatrix = data.corr()
mask = np.triu(np.ones_like(corrMatrix, dtype=bool))

plt.figure(figsize=(20,15))
sns.heatmap(corrMatrix, mask=mask, cmap=sns.diverging_palette(220, 10, as_cmap=True), 
            annot=True, linewidths=2, fmt = ".2f")
plt.title("Coorelation Heatmap")
plt.show()

#Feature Selection

##Univariate Feature Selection
#This will help us to get a more structured and correlated model for training and testing
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

# print(XNew.info())

#Spliting Data
XTrain, XTest, yTrain, yTest = train_test_split(XNew, y, test_size=0.35, random_state=1)

#Model Training and fitting
svcModel = SVC()
svcModel.fit(XTrain, yTrain)

#Model Evalulation

#Without Data Normalization
yPred = svcModel.predict(XTest)
confMatrix = confusion_matrix(yTest, yPred)
plt.figure(figsize=(8, 6))
sns.heatmap(confMatrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print(classification_report(yTest, yPred))

#With Data Normalization 
#This will help to bring down the values to [0,1]

scaler = MinMaxScaler()

XTrainScaled = scaler.fit_transform(XTrain.astype(np.float_))
XTestScaled = scaler.fit_transform(XTest.astype(np.float_))

svcModel.fit(XTrainScaled, yTrain)
yPredScaled = svcModel.predict(XTestScaled)

scaledConfmatrix = confusion_matrix(yTest, yPredScaled)
plt.figure(figsize=(8, 6))
sns.heatmap(confMatrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print(classification_report(yTest, yPredScaled))

#With Data Normalization, we get increased accuracy and weighted avg
