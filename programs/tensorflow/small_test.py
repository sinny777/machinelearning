
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics

# LOAD DATA
# df = pd.read_csv('../../data/raw_car_dashboard.csv', sep=',')
df = pd.read_csv('../../data/raw_home_automation.csv', sep=',')

# print(df.describe())

# df_ = df.groupby('domain')['ID'].nunique()
# print(df["intent"].value_counts())

X_train = df['utterances']
Y_train = df['intent']

X_test = pd.Series(["how is the weather in Kanpur", "turn on the living room light"])
Y_test = pd.Series(["weather", "appliance_action"])

# print(df.head())

from sklearn.pipeline import Pipeline
clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
clf = clf.fit(X_train, Y_train)
predicted = clf.predict(X_test)
print("Prediction using Pipeline: >> ", predicted)
print("Performance of MultinomialNB Classifier: >> ", np.mean(predicted == Y_test))
