
#!/usr/bin/env python

"""Functions for downloading and reading MNIST data."""
import pandas as pd
import numpy as np
import random

import os.path
from os import path

import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
from nltk.cluster.util import cosine_distance
from nltk import word_tokenize,sent_tokenize,ne_chunk
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

def convert_to_predict(filename, sentence):
    df = pd.read_csv(filename)
    # df.head()
    classes = []
    documents = []
    words = []
    ignore_words = ['?']
    # loop through each sentence in our intents patterns
    for i in range(len(df)):
        # tokenize each word in the sentence
        w = nltk.word_tokenize(df["utterances"][i])
        # add to our words list
        words.extend(w)
        # add to documents in our corpus
        documents.append((w, df["intent"][i]))
        # add to our classes list
        if df["intent"][i] not in classes:
            classes.append(df["intent"][i])

    # stem and lower each word and remove duplicates
    words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))
    to_predict = bow(sentence, words)
    return to_predict, classes

def extract_training_data(filename):
    df = pd.read_csv(filename)
    # df.head()
    classes = []
    documents = []
    words = []
    ignore_words = ['?']
    # loop through each sentence in our intents patterns
    for i in range(len(df)):
        # tokenize each word in the sentence
        w = nltk.word_tokenize(df["utterances"][i])
        # add to our words list
        words.extend(w)
        # add to documents in our corpus
        documents.append((w, df["intent"][i]))
        # add to our classes list
        if df["intent"][i] not in classes:
            classes.append(df["intent"][i])

    # stem and lower each word and remove duplicates
    words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))
    # remove duplicates
    classes = sorted(list(set(classes)))
    print (len(documents), "documents")
    print (len(classes), "classes", classes)

    training = []
    output = []
    # create an empty array for our output
    output_empty = [0] * len(classes)
    # training set, bag of words for each sentence
    for doc in documents:
        bag = []
        pattern_words = doc[0]
        pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)

        # output is a '0' for each tag and '1' for current tag
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1

        training.append([bag, output_row])

    # shuffle our features and turn into np.array
    random.shuffle(training)
    training = np.array(training)
    return training

class DataSet(object):
    def __init__(self, intents, classes):
        # assert intents.shape[0] == classes.shape[0], (
        #     "intents.shape: %s classes.shape: %s" % (intents.shape,
        #                                            classes.shape))
        # self._num_examples = intents.shape[0]
            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            # assert intents.shape[3] == 1
            # intents = intents.reshape(intents.shape[0],
            #                         intents.shape[1] * intents.shape[2])
            # # Convert from [0, 255] -> [0.0, 1.0].
            # intents = intents.astype(numpy.float32)
            # intents = numpy.multiply(intents, 1.0 / 255.0)
        self._intents = intents
        self._classes = classes
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def intents(self):
        return self._intents

    @property
    def classes(self):
        return self._classes

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._intents = self._intents[perm]
            self._classes = self._classes[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._intents[start:end], self._classes[start:end]


def read_data_sets(intents_file):
    class DataSets(object):
        pass
    data_sets = DataSets()
    INTENTS = intents_file
    VALIDATION_SIZE = 3
    training = extract_training_data(INTENTS)
    # train_x = list(training[:,0])
    # train_y = list(training[:,1])
    train_x = list(training[:,0])
    train_y = list(training[:,1])
    data_sets.train = DataSet(train_x, train_y)
    # data_sets.train = DataSet(train_x[VALIDATION_SIZE:], train_y[VALIDATION_SIZE:])
    # data_sets.validation = DataSet(train_x[:VALIDATION_SIZE], train_y[:VALIDATION_SIZE])
    # data_sets.test = DataSet(test_images, test_labels)
    return data_sets
