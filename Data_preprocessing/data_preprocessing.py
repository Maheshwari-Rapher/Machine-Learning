import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

dataset = pd.read_csv(r"C:\Users\Maheshwari\Downloads\ML_NIT\data preprocessing\Data.csv")

#divide the dataset into x and y
x = dataset.iloc[:,:-1].values

y= dataset.iloc[:,3].values


# fill the missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer()

imputer = imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])


#Categorical data to numerical data 
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
labelencoder_x.fit_transform(x[:,0])
x[:,0]=labelencoder_x.fit_transform(x[:,0])


labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Split the dataset into x train , x-test and y- train and y-test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y , train_size=0.7 ,random_state=0)




