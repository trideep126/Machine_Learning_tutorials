#Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#Generate Sample Data
np.random.seed(0)
X = np.sort(np.random.uniform(0,1,20))
y = np.sin(2 * np.pi * X) + np.random.normal(0,1,20)

#Fit the polyonomial regression model
degree = 3
poly_features = PolynomialFeatures(degree = degree,include_bias = False)
X_poly = poly_features.fit_transform(X.reshape(-1,1))

model = LinearRegression()
model.fit(X_poly,y)

#Generate points for plotting polynomial curve
X_plot = np.linspace(0,1,100)
X_plot_poly = poly_features.transform(X_plot.reshape(-1,1))
y_plot = model.predict(X_plot_poly)

#Plotting the data and polynomial curve
plt.figure(figsize=(8,6))
plt.scatter(X,y,label='Data Labels')
plt.plot(X_plot,y_plot,color='red',label=f'Polynomial Regression (Degree {degree})')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)