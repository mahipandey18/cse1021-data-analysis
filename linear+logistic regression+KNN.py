# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 12:39:03 2024

@author: DELL
"""

import pandas as pd 
from sklearn.metrics import accuracy_score
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn import preprocessing , svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import random as rn 
from sklearn.metrics import mean_squared_error

data = pd.read_csv('world-happiness-report-2021.csv')
print(data.describe)
#data type
print(data.info())
print(data.describe())
print(data.mean)

data1 = data[['Country name', 'Regional indicator', 'Ladder score',
        'Logged GDP per capita', 'Social support', 'Healthy life expectancy',
       'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption']]

cols = data1.corr()['Ladder score'].sort_values(ascending=False)

fig = plt.figure(figsize=(15,10))
plt.suptitle("Comparing the features that contribute for Ladder score",family='Serif', weight='bold', size=20)
plt.legend()
plt.show()

X = data[["Ladder score"]]
Y = (data["Healthy life expectancy"] >= 5).astype(int)
 
#KNN model
sns.scatterplot(data=data)
plt.plot(X,Y)
plt.show()

rn.seed(1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=1)

model = KNeighborsClassifier(n_neighbors=5) 
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(Y_test, Y_pred)}")

#linear regression model
model1 = LinearRegression()
model1.fit(X_train, Y_train)

Y_pred = model1.predict(X_test)

mse = mean_squared_error(Y_test, Y_pred)
print(f"Mean Squared Error: {mse}")

ladder_score = np.linspace(data["Ladder score"].min(), data["Ladder score"].max(), 100).reshape(-1, 1)
prediction = model1.predict(ladder_score)

plt.scatter(data["Ladder score"], data["Healthy life expectancy"], color='blue', label='Actual Data')

plt.plot(ladder_score, prediction, color='red', label='Regression Line')

plt.xlabel("Ladder score")
plt.ylabel("Healthy life expectancy")
plt.title('Linear Regression: ladder score vs life expectancy')
plt.legend()
plt.grid()
plt.show()

# Train a logistic regression model
model2 = LogisticRegression()
model2.fit(X_train, Y_train)

Y_pred = model2.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy}")

ladder_score1 = np.linspace(data["Ladder score"].min(), data["Ladder score"].max(), 100).reshape(-1, 1)
prediction1 = model2.predict(ladder_score1)

plt.scatter(data["Ladder score"], data["Healthy life expectancy"], color='blue', label='Actual Data')

plt.plot(ladder_score1, prediction1, color='red', label='Logistic Regression Boundary')

plt.xlabel('Ladder score')
plt.ylabel('Healthy life expectancy')
plt.title('Logistic Regression: ladder score vs life expectancy')
plt.legend()
plt.grid()

plt.show()






















