# LGBM 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Read the dataset
dataset = pd.read_csv(r"C:\Users\Maheshwari\Downloads\ML_NIT\Classification_nit\TREE algorithms\Ensamble learning\Brust cancer project\Brust cancer.csv")

# divide the dataset intox and y
X = dataset.iloc[:,0:5].values
y = dataset.iloc[:, -1].values

# split the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# Build the lightgbm model
import lightgbm as lgb
clf = lgb.LGBMClassifier()
clf.fit(X_train,y_train)


# Predicting the Test set results
y_pred = clf.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# Model Accuracy 
from sklearn.metrics import accuracy_score 
ac = accuracy_score(y_test, y_pred)
print(ac)


# This is to get the Classification Report
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
cr

bias = clf.score(X_train,y_train)
bias

variance = clf.score(X_test,y_test)
variance



























