# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# read theg dataset
dataset=pd.read_csv(r"C:\Users\Maheshwari\Downloads\ML_NIT\Classification_nit\logit classification.csv")

# Divide the x and y
x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values

# split the dataset into training and testing
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# Build the  Logistic Regressor model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train,y_train)


#predicting the test set
y_pred = classifier.predict(x_test)


# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# making the accuracy
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)


from sklearn.metrics import classification_report
cr = classification_report(y_test,y_pred)
print(cr)

# bias 
bias = classifier.score(x_train,y_train)
print(bias)

#variance 
variance = classifier.score(x_test,y_test)
print(variance)

#------------------FUTURE PREDICTION--------------------
dataset1 = pd.read_csv(r"C:\Users\Maheshwari\Downloads\ML_NIT\Classification_nit\final1.csv")

d2 = dataset1.copy()

dataset1 = dataset1.iloc[:, [2,3]].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
M = sc.fit_transform(dataset1)


#prediction
y_pred1 = pd.DataFrame()

d2['y_pred1'] = classifier.predict(M)

d2.to_csv('final1.csv')



















