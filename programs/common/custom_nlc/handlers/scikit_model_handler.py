
#!/usr/bin/env python

import pandas as pd
import numpy as np
import random

import os.path
from os import path
import pickle
from sklearn.externals import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer

from handlers.data_handler import DataHandler, DataSet

class ModelHandler(object):
    def __init__(self, data_handler, CONFIG):
        self.data_handler = data_handler
        self.CONFIG = CONFIG
        self.ensure_dir(self.CONFIG["MODEL_PATH"])
        class DataSets(object):
            pass
        self.datasets = DataSets()
        self.prepare_data()

    def ensure_dir(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def prepare_data(self):
        training = self.data_handler.get_training_data()
        train_x = list(training[:,0])
        train_y = list(training[:,1])
        print("Training Data Length: ", len(train_x))
        print("Training Data Target Length: ", len(train_y))
        self.datasets.train = DataSet(train_x, train_y)

    def create_model(self, PARAMS):
        global model
        model = RandomForestClassifier(n_estimators = PARAMS["n_estimators"])
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
