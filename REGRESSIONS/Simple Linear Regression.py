import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read the dataset
dataset = pd.read_csv(r"C:\Users\Maheshwari\OneDrive\Desktop\Machine-learning work\REGRESSIONS\simple_linear_regression\Salary_Data.csv")

# split the dataset into x and y
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

# again split into x-train ,x -test and y-train,y-test
from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(x, y,test_size=0.20,random_state=0)


# Built the model by using x_train and y_train
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# test the model by passing the x_test
y_pred = regressor.predict(x_test)

#comparing the actual and predicted salaries from the test set
comparision = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(comparision)


# visualize the test set
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train,regressor.predict(x_train),color = 'blue')
plt.title('Salary vs Experience(Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# slope
m_slope = regressor.coef_
print(m_slope)

c_intercept = regressor.intercept_
print(c_intercept)

#prediction 1 :
y_12 = (m_slope*12) + c_intercept
print(y_12)

#prediction 2 :
y_20 = (m_slope*20) + c_intercept
print(y_20)

#implementing the statistics
# Mean
dataset.mean()
dataset['Salary'].mean()

#Median
dataset.median()
dataset['Salary'].median()

# Mode
dataset.mode()
dataset['Salary'].mode()

# Variance
dataset.var()

# Standard deviation
dataset.std()
dataset['Salary'].std()

# Coefficient of variation
from scipy.stats import variation
variation(dataset.values)
variation(dataset['Salary'])

# Correlation
dataset.corr()
dataset['Salary'].corr(dataset['YearsExperience'])

# Skewness
dataset.skew()
dataset['Salary'].skew()

# Standard Error
dataset.sem()
dataset['Salary'].sem()

# Z-score 
import scipy.stats as stats
dataset.apply(stats.zscore)
stats.zscore(dataset['Salary'])

# Degree of Freedom
a = dataset.shape[0]   #this will give no.of rows
b = dataset.shape[1]   #this will give no.of columns

degree_of_freedom = a - b
print(degree_of_freedom)

# Anova
y_mean = np.mean(y)

# SSR (Sum of Squared Regression)
SSR = np.sum((y_pred - y_mean)**2)
print(SSR)

#SSE ( Sum of Square error)
y = y[0:6]
SSE = np.sum((y - y_pred)**2)
print(SSE)

# SST (Sum of square total)
SST = SSR + SSE
print(SST)

# R- square
r_square = 1 - (SSR / SST)
print(r_square)

# bias (training)
bias = regressor.score(x_train,y_train)
print(bias)

#variance  (testing)
variance = regressor.score(x_test,y_test)
print(variance)


from sklearn.metrics import mean_squared_error
train_mse = mean_squared_error(y_train,regressor.predict(x_train))
test_mse = mean_squared_error(y_test , y_pred)

# Frontend
import pickle

#save the trained model to disk
filename = 'linear_regression_model.pkl'

#open a file in write-binary mode and dump the model
with open(filename,'wb') as file:
    pickle.dump(regressor,file)
    
print("Model has been pickled and saved as linear_regression_model.pkl")    


import os
os.getcwd()

