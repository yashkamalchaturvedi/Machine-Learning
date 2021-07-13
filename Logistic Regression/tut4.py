# Train Logistic Regression Classifier to Predict Whether a Flower is Iris Virginica or not

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
# print(list(iris.keys())) # ['data', 'target', 'target_names', 'DESCR', 'feature_names']
# print(iris['data'])
# print(iris['data'].shape) # (150, 4) Rows. Columns
# print(iris['target'])
# print(iris['DESCR'])

X = iris["data"][:, 3:]  # Slicing only 3rd Column Rows as it is
# Copy of the array, cast to a specified type.
y = (iris["target"] == 2).astype(np.int)

# Train a Logistic Regression Classifier
clf = LogisticRegression()
clf.fit(X, y)
example = clf.predict(([[2.6]]))  # Predict class labels for samples in X.
# print(example)

# Using Matplotlib to Plot the Visualization
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
# linspace Return evenly spaced numbers over a specified interval. Here 100 points between 0 and 3.
# reshape Gives a new shape to an array without changing its data. Here it makes 1D Array.
# Probability estimates. The returned estimates for all classes are ordered by the label of classes.
y_prob = clf.predict_proba(X_new)
print(y_prob)
plt.plot(X_new, y_prob[:, 1], "g-", label="virginica")  # gives 1 row
plt.show()
