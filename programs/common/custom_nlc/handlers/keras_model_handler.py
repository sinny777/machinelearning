
#!/usr/bin/env python

import pandas as pd
import numpy as np
import random

import os.path
from os import path

import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Input, concatenate, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
# from tf.keras.layers.pooling import GlobalMaxPooling1D, MaxPooling1D
# from tf.keras.layers.core import Dropout
from keras import backend as K
from keras.models import load_model

from handlers.data_handler import DataHandler, DataSet

class ModelHandler(object):
    def __init__(self, data_handler, CONFIG):
        self.name = "keras"
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
        X, Y = self.data_handler.get_training_data()
        print("Training Data Length: ", len(X))
        print("Training Data Target Length: ", len(Y))
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.10, random_state = 42)
        self.datasets.train = DataSet(X_train, Y_train)
        self.datasets.test = DataSet(X_test, Y_test)

    def create_model(self, PARAMS):
        K.clear_session()
        tf.reset_default_graph()
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        with tf.Session() as sess:
          sess.run(init_g)
          sess.run(init_l)
          # Create the network definition based on Gated Recurrent Unit (Cho et al. 2014).
          embedding_vector_length = 32

          model = Sequential()
          model.add(Embedding(self.data_handler.max_features, embedding_vector_length, input_length=self.data_handler.maxlen))
          model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
          model.add(MaxPooling1D(pool_size=2))
          model.add(LSTM(100))
          model.add(Dense(len(self.datasets.train.intents[0]), activation=PARAMS["activation"]))
          model.compile(loss=PARAMS["loss"], optimizer=PARAMS["optimizer"], metrics=PARAMS["metrics"])
          print(model.summary())

          tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=PARAMS["log_dir"], write_graph=True)

          monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=PARAMS["patience"], verbose=0, mode='auto')
          checkpointer = ModelCheckpoint(filepath=self.CONFIG["MODEL_WEIGHTS_PATH"], verbose=0, save_best_only=True) # Save best model
          model.fit(self.datasets.train.utterances, self.datasets.train.intents, epochs=PARAMS["epochs"], batch_size=PARAMS["batch_size"],  verbose=1, validation_split=0.05, callbacks=[tbCallBack, monitor, checkpointer])
          model.load_weights(self.CONFIG["MODEL_WEIGHTS_PATH"]) # load weights from best model
          scores = model.evaluate(self.datasets.test.utterances, self.datasets.test.intents)
          print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
          model.save(self.CONFIG["MODEL_PATH"])
          print("<<<<<<<< ML MODEL CREATED AND SAVED >>>>>>>>>>>\n\n")

    def load_keras_model(self):
        model = load_model(self.CONFIG["MODEL_PATH"])
        model.load_weights(self.CONFIG["MODEL_WEIGHTS_PATH"]) # load weights from best model
        return model

    def predict(self, text):
        ERROR_THRESHOLD = 0.15
        model = self.load_keras_model()
        toPredict = self.data_handler.convert_to_predict(text)
        predictions = model.predict(np.array(toPredict))[0]
        # np.argmax(predictions[0])
        # filter out predictions below a threshold
        predictions = [[i,r] for i,r in enumerate(predictions) if r>ERROR_THRESHOLD]
        # sort by strength of probability
        predictions.sort(key=lambda x: x[1], reverse=True)
        print("predictions: >> ", predictions)
        return_list = []
        for r in predictions:
            return_list.append((self.data_handler.intents[r[0]], r[1]))
        # return tuple of intent and probability
        return return_list
