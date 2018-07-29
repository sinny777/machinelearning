from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

iris = load_iris()
print("Feature Names: %s" % iris.feature_names)


# print(iris.data)
# print(iris.target)
print(iris.data.shape)
print(iris.target.shape)

X = iris.data
y = iris.target

# print(y)

clf = KNeighborsClassifier(n_neighbors = 1)
# clf = LogisticRegression()
clf.fit(X, y)
y_pred = clf.predict(X)
# print("y_pred using Nearest Neighbor - KNN: >> ")
# print(y_pred);

print(metrics.accuracy_score(y, y_pred));
