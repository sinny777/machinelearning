
#!/usr/bin/env python

"""Functions for downloading and reading MNIST data."""
import pandas as pd
import numpy as np
import random
import re
import json

import os.path
from os import path

# from utilities.NLPUtility import NLPUtility

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class DataHandler(object):
    def __init__(self, dataframe, library_name):
        self.dataframe = dataframe
        self.library_name = library_name
        self.intents = []
        self.max_features = 500
        self.maxlen = 50
        self.tokenizer = None

    def get_tokenizer(self):
        if self.tokenizer:
            pass
        else:
            self.tokenizer = Tokenizer(num_words=self.max_features, split=' ')
            self.tokenizer.fit_on_texts(self.dataframe["utterances"].values)
        return self.tokenizer

    def get_training_data(self):
        documents = []
        X = self.get_tokenizer().texts_to_sequences(self.dataframe["utterances"].values)
        X = pad_sequences(X, maxlen=self.maxlen)
        self.intents = self.dataframe["intent"].unique()
        self.intents = sorted(list(set(self.intents)))
        output_empty = [0] * len(self.intents)
        Y = []
        for i in range( 0, len(X)):
            output_row = list(output_empty)
            intent = self.dataframe["intent"][i]
            output_row[self.intents.index(intent)] = 1
            if self.library_name == 'scikit':
                Y.append(intent)
            else:
                Y.append(output_row)

        print("Length of X: >> ", len(X))
        print("Length of Y: >> ", len(Y))
        return X, Y

    def get_intents(self):
        documents = []
        self.intents = self.dataframe["intent"].unique()
        self.intents = sorted(list(set(self.intents)))
        return self.intents

class DataSet(object):
    def __init__(self, utterances, intents):
        self._utterances = utterances
        self._intents = intents

    @property
    def utterances(self):
        return self._utterances

    @property
    def intents(self):
        return self._intents
