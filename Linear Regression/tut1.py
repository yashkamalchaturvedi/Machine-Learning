# Linear Regression

import matplotlib.pyplot as plt
# (module) plt
# This is an object-oriented plotting library.
# A procedural interface is provided by the companion pyplot module, which may be imported directly, e.g.:
# import matplotlib.pyplot as plt
# matplotlib was initially written by John D. Hunter (1968-2012) and is now developed and maintained by a host of others.

import numpy as np
# (module) np NumPy Provides
# An array object of arbitrary homogeneous items
# Fast mathematical operations over arrays
# Linear Algebra, Fourier Transforms, Random Number Generation

from sklearn import datasets, linear_model
# The sklearn.datasets module includes utilities to load datasets, including methods to load and fetch popular reference datasets.
# It also features some artificial data generators.

from sklearn.metrics import mean_squared_error
# Mean squared error regression loss

diabetes = datasets.load_diabetes()
# Load and return the diabetes dataset (regression).

# print(diabetes.keys())
# D.keys() -> a set-like object providing a view on D's keys
# dict_keys(['data', 'target', 'DESCR', 'feature_names'])

# print(diabetes.data)
# print(diabetes.DESCR)

# diabetes_X = diabetes.data[:, np.newaxis, 2]
# # For 1 feature 1 label
# Mean squared error is:  3035.0601152912695
# Weights:  [941.43097333]
# Intercept:  153.39713623331698

diabetes_X = diabetes.data
# For all features but no plotting
# Mean squared error is:  1826.5364191345423
# Weights:  [  -1.16924976 -237.18461486  518.30606657  309.04865826 -763.14121622
#   458.90999325   80.62441437  174.32183366  721.49712065   79.19307944] (w1,w2,w3,..)
# Intercept:  153.05827988224112 (wo)

diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]

diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-30:]

model = linear_model.LinearRegression()
# The sklearn.linear_model module implements generalized linear models.
# It includes Ridge regression, Bayesian Regression, Lasso and Elastic Net estimators computed with Least Angle Regression and coordinate descent.
# It also implements Stochastic Gradient Descent related algorithms.

model.fit(diabetes_X_train, diabetes_y_train)
# Fit linear model

diabetes_y_predicted = model.predict(diabetes_X_test)
# Predict using the linear model

print("Mean squared error is: ", mean_squared_error(
    diabetes_y_test, diabetes_y_predicted))

print("Weights: ", model.coef_)
print("Intercept: ", model.intercept_)

# plt.scatter(diabetes_X_test, diabetes_y_test)
# plt.plot(diabetes_X_test, diabetes_y_predicted)

# plt.show()
# Display a figure