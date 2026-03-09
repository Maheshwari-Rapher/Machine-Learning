#  Grid search CV

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r"C:\Users\Maheshwari\Downloads\ML_NIT\Classification_nit\KNN\Social_Network_Ads.csv")

# divide the dataset intox and y
X = dataset.iloc[:,2:4].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,random_state = 0)

# Training the  kernel SVM model on the Training set
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

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

bias = classifier.score(X_train,y_train)
bias

variance = classifier.score(X_test,y_test)
variance

'''
# Re-train
# Applying K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X = X_train , y = y_train, cv = 40)
print(accuracies.mean()*100)
'''

# Applying the GRID search to find the best model and best parameter
from sklearn.model_selection import GridSearchCV
parameters = [{'C' : [1,10,100,1000],'kernel':['linear']},
              {'C': [1,10,100,1000],'kernel':['rbf'],'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7]}]
grid_search = GridSearchCV(estimator= classifier,
                            param_grid = parameters,
                            scoring = 'accuracy',
                            cv = 10,
                            )
grid_search = grid_search.fit(X_train,y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy:{:2f} %".format(best_accuracy*100))
print("Best Parameters :", best_parameters)                           
                           












































































