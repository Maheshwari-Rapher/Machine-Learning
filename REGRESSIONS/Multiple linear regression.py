# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read the dataset
dataset= pd.read_csv(r"C:\Users\Maheshwari\OneDrive\Desktop\Machine-learning work\REGRESSIONS\Multiple_linear_regression\Investment.csv")


# Splitting the data
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4]


# convert categorical to numerical values
x = pd.get_dummies(x,dtype=int)

# select the model
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test  = train_test_split(x,y, test_size=0.2,random_state=0)


# bulid the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


# prediction
y_pred = regressor.predict(x_test)

bias = regressor.score(x_train,y_train)
bias

variance = regressor.score(x_test,y_test)
variance

#slope
m_slope = regressor.coef_
m_slope


# Intercept
c_intercept = regressor.intercept_
print(c_intercept)


# add the constant to each row
x = np.append(arr = np.ones((50,1)).astype(int),values = x, axis=1)


import statsmodels.api as sm
x_opt = x[:,[0,1,2,3,4,5]]
# ordinary least square
regressor_OLS = sm.OLS( endog = y, exog = x_opt).fit()
regressor_OLS.summary()


import statsmodels.api as sm
x_opt = x[:,[0,1,2,3,5]]
# ordinary least square
regressor_OLS = sm.OLS( endog = y, exog = x_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
x_opt = x[:,[0,1,2,3]]
# ordinary least square
regressor_OLS = sm.OLS( endog = y, exog = x_opt).fit()
regressor_OLS.summary()


import statsmodels.api as sm
x_opt = x[:,[0,1,3]]
# ordinary least square
regressor_OLS = sm.OLS( endog = y, exog = x_opt).fit()
regressor_OLS.summary()


import statsmodels.api as sm
x_opt = x[:,[0,1]]
# ordinary least square
regressor_OLS = sm.OLS( endog = y, exog = x_opt).fit()
regressor_OLS.summary()



























