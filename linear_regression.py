# Linear Regression

## Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import numpy as np

## Read the data
d = pd.read_csv('ex1data1.txt', header=None, names=['city_pop', 'profit'])
y = d.profit    #define dependant variable
print(y.head())
print(d.head())

## Plot the data
sns.scatterplot(d['city_pop'], d['profit'])
plt.show()

## Create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(d.city_pop, y, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

X_train = X_train.to_numpy().reshape((-1, 1))
print(X_train.shape)

X_test = X_test.to_numpy().reshape((-1, 1))
print(X_test.shape)

## Fit the model
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)

print(predictions[0:5])
print(type(predictions)) 

sns.scatterplot(y_test, predictions)
plt.show()

print('Score:', model.score(X_test, y_test))
print('Intercept:', model.intercept_)
print('Slope:', model.coef_)
