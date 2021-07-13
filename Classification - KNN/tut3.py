# Loading Required Modules
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# Loading Datasets
iris = datasets.load_iris()
# Load and return the iris dataset (classification).
# The iris dataset is a classic and very easy multi-class classification dataset.

# Printing Description and Features
print(iris.DESCR)
features = iris.data
labels = iris.target
print(features[0], labels[0])

# Training a Classifier
clf = KNeighborsClassifier()
clf.fit(features, labels)

preds = clf.predict([[1, 1, 1, 1]])
# Predict the class labels for the provided data
print(preds)