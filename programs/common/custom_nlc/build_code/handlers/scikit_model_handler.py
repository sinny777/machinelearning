
#!/usr/bin/env python

import pandas as pd
import numpy as np
import random

import os.path
from os import path
import pickle
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer

from build_code.handlers.data_handler import DataHandler, DataSet

class ModelHandler(object):
    def __init__(self, CONFIG):
        self.name = "scikit"
        self.CONFIG = CONFIG
        self.data_handler = self.get_data_handler()
        class DataSets(object):
            pass
        self.datasets = DataSets()
        self.prepare_data()

    def get_data_handler(self):
        df = pd.read_csv(self.CONFIG["DATA_FILE_PATH"], header=0, delimiter=",")
        return DataHandler(df, "scikit")

    def prepare_data(self):
        X, Y = self.data_handler.get_training_data()
        print("Training Data Length: ", len(X))
        print("Training Data Target Length: ", len(Y))
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.10, random_state = 42)
        self.datasets.train = DataSet(X_train, Y_train)
        self.datasets.test = DataSet(X_test, Y_test)

    def create_model(self):
        global model
        model = RandomForestClassifier(n_estimators = self.CONFIG["MODEL_CONFIG"]["epochs"])
        # model = MultinomialNB()
        model = model.fit(self.datasets.train.utterances, self.datasets.train.intents)
        saved_model = joblib.dump(model, self.CONFIG["MODEL_PATH"])
        print("<<<<<<<< ML MODEL CREATED AND SAVED >>>>>>>>>>>\n\n")
        return model

    def load_model(self):
        return joblib.load(self.CONFIG["MODEL_PATH"])

    def predict(self, text):
        toPredict = self.data_handler.convert_to_predict(text)
        model = self.load_model()
        results = model.predict(toPredict)
        return results
