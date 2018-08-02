
#!/usr/bin/env python

"""Functions for downloading and reading MNIST data."""
import pandas as pd
import numpy as np
import random

import os.path
from os import path

from utilities.NLPUtility import NLPUtility

class DataHandler(object):
    def __init__(self, dataframe, library_name):
        self.dataframe = dataframe
        self.library_name = library_name
        self.processed_words = []
        self.intents = []


    def get_training_data(self):
        # Initialize an empty list to hold the clean text
        documents = []
        ignore_words = ['?']
        print("Cleaning and parsing the training set movie reviews...\n")
        for i in range( 0, len(self.dataframe["utterances"])):
            # w = NLPUtility.text_to_wordlist(self.dataframe["utterances"][i], True, True)
            # w = NLPUtility.text_to_word_sequence(self.dataframe["utterances"][i])
            w = NLPUtility.tokenize_sentence(self.dataframe["utterances"][i])
            self.processed_words.extend(w)
            documents.append((w, self.dataframe["intent"][i]))
            if self.dataframe["intent"][i] not in self.intents:
                self.intents.append(self.dataframe["intent"][i])

        # remove duplicates
        self.intents = sorted(list(set(self.intents)))

        print("Total Processed Words Before Sorting: ", len(self.processed_words))
        self.processed_words = NLPUtility.stem_words(self.processed_words, ignore_words)
        self.processed_words = sorted(list(set(self.processed_words)))
        print("Total Processed Words After Sorting: ", len(self.processed_words))
        print("Total Utterances: ", len(documents))
        print("Total Intents: ", len(self.intents))

        training = []
        # create an empty array for our output
        output_empty = [0] * len(self.intents)
        # training set, bag of words for each sentence
        for doc in documents:
            # initialize our bag of words
            bag = []
            pattern_words = doc[0]
            ignore_words = ['?']
            pattern_words = NLPUtility.stem_words(pattern_words, ignore_words)
            for w in self.processed_words:
                bag.append(1) if w in pattern_words else bag.append(0)

            if self.library_name == 'scikit':
                training.append([bag, doc[1]])
            else:
                # output is a '0' for each tag and '1' for current tag
                output_row = list(output_empty)
                output_row[self.intents.index(doc[1])] = 1
                training.append([bag, output_row])


        # shuffle our features and turn into np.array
        random.shuffle(training)
        training = np.array(training)
        return training

    def convert_to_predict(self, text):
        # sentence_words = NLPUtility.text_to_wordlist(text, True, True)
        # sentence_words = NLPUtility.text_to_word_sequence(text)
        sentence_words = NLPUtility.tokenize_sentence(text)
        bag = []
        for w in self.processed_words:
            bag.append(1) if w in sentence_words else bag.append(0)
        return np.array(bag)

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
