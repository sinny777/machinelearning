
import os.path
from os import path
from io import  StringIO
import requests
import json
from datetime import datetime
import time
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics

import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

# LOAD DATA
df = pd.read_csv('../../data/raw_car_dashboard.csv', sep=',')

print(df.describe())

X_train = df['utterances']
Y_train = df['intent']

X_test = pd.Series(["Is it going to rain next friday", "how far is office from here"])
Y_test = pd.Series(["weather", "navigation"])

# print(df.head())

from sklearn.pipeline import Pipeline
clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
clf = clf.fit(X_train, Y_train)
predicted = clf.predict(X_test)
print("Prediction using Pipeline: >> ", predicted)
print("Performance of MultinomialNB Classifier: >> ", np.mean(predicted == Y_test))
