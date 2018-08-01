import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets, metrics

iris = datasets.load_iris()
# print(iris.target)

X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.3)

# classifier = LogisticRegression()
classifier = GaussianNB()
# classifier = KNeighborsClassifier(n_neighbors = 1)
classifier.fit(X_train, Y_train)
score = metrics.accuracy_score(Y_test, classifier.predict(X_test))
print("Accuracy: %f" % score)
