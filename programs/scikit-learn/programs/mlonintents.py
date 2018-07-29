import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics

data = pd.read_csv('intents_data.csv', sep=',')
for label in ['intent', 'class']:
    data[label] = LabelEncoder().fit_transform(data[label])

# Take the fields of interest and plug them into variable X
X = data[['intent']]
print('X: >> ', X)
# Make sure to provide the corresponding truth value
y = data['class'].values.tolist()
print(np.array(y))

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
# tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
# X_train_tf = tf_transformer.transform(X_train_counts)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train)
print(X_train_tfidf.shape)

# classifier = LogisticRegression()
# classifier = GaussianNB()
# classifier = KNeighborsClassifier(n_neighbors = 1)
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, Y_train)

# docs_new = ['whats the weather outside', 'it is cloudy now']
X_new_counts = count_vect.transform(X_test)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
toPredict = classifier.predict(X_new_tfidf)
print(toPredict)

# print(classifier.predict(toPredict))

score = metrics.accuracy_score(Y_test, classifier.predict(X_test))
print("Accuracy: %f" % score)
