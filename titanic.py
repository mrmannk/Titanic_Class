import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
# load the dataset, change paths according to your file location
titanic_train = pd.read_csv('train.csv') 
titanic_test = pd.read_csv('test.csv')
# first 5 rows of the dataset
titanic_train.head()
# no of rows and columns
print (titanic_train.shape)
print (titanic_train["Survived"].value_counts())
# only the distinct values
titanic_train["Survived"].value_counts().keys()
# size of graph
plt.figure(figsize=(5,5))
plt.bar(list(titanic_train["Survived"].value_counts().keys()),list(titanic_train["Survived"].value_counts()),color=["r","g"])
plt.title("Distribution of Survival")  
plt.show()
#passanger class
print (titanic_train['Pclass'].value_counts())
print (titanic_train["Pclass"].value_counts().keys())
plt.figure(figsize=(5,5))
plt.bar(list(titanic_train["Pclass"].value_counts().keys()),list(titanic_train["Pclass"].value_counts()),color=["r","g","b"])  
plt.title("Distribution of Passanger Class")
plt.show()
# gender
print (titanic_train["Sex"].value_counts())
print (titanic_train["Sex"].value_counts().keys())
plt.figure(figsize=(5,5))
plt.bar(list(titanic_train["Sex"].value_counts().keys()),list(titanic_train["Sex"].value_counts()),color=["r","g"])
plt.title("Gender Distribution")  
plt.show()
# age
plt.figure(figsize=(5,7))
plt.hist(titanic_train["Age"],color="orange")
plt.title("Distribution of Age")    
plt.xlabel("Age")
plt.show()
# checking count of null values
sum(titanic_train["Survived"].isnull())
sum(titanic_train["Age"].isnull())
# to remove null values
titanic_train = titanic_train.dropna()
# cross checking
print (sum(titanic_train["Survived"].isnull()))
print (sum(titanic_train["Age"].isnull()))
# assigning variables
x_train = titanic_train[['Age']]
y_train = titanic_train[['Survived']]
# fitting the model	
dtc.fit(x_train,y_train)
# checking count of null values
sum(titanic_test["Age"].isnull())
# to remove null values
titanic_test = titanic_test.dropna()
# cross checking
print (sum(titanic_test["Age"].isnull()))
# assigning variables
x_test = titanic_test[['Age']]
# predicting
y_pred = dtc.predict(x_test)
print (y_pred)
