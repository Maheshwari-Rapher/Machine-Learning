# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#read the dataset
dataset = pd.read_csv(r"C:\Users\Maheshwari\Downloads\ML_NIT\Regression_nit\NON LINEAR REGRESSION MODEL\Polynomial linear regression\emp_sal.csv")

# split into x and y
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#build simple linear regression  (degree - 1)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)


# Build the visualization
plt.scatter(x, y, color = 'red')
plt.plot(x,lin_reg.predict(x),color = 'blue')
plt.title('Linear regression model(Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#prediction
lin_model_pred = lin_reg.predict([[6]])
lin_model_pred

# Build polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
x_poly = poly_reg.fit_transform(x)

poly_reg.fit(x_poly , y)

#again build the simple linear model
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)

# visualize the lin_reg_2
plt.scatter(x, y, color = 'red')
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)),color='blue')
plt.title('Polymodel ( Polynomial  Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# polynomial predicts
poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[7]]))
poly_model_pred
































