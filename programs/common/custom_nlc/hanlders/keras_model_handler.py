
#!/usr/bin/env python

import pandas as pd
import numpy as np
import random

import os.path
from os import path

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Input, concatenate, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
# from tf.keras.layers.pooling import GlobalMaxPooling1D, MaxPooling1D
# from tf.keras.layers.core import Dropout
from keras import backend as K
from keras.models import load_model

from hanlders.data_handler import DataHandler, DataSet

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
        K.clear_session()
        tf.reset_default_graph()
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        with tf.Session() as sess:
          sess.run(init_g)
          sess.run(init_l)

          model = Sequential()
          # model.add(Dense(output_dim=8,init='uniform',activation='relu', input_dim=len(train_x[0])))
          model.add(Dense(PARAMS["batch_size"], activation='relu', input_shape=(np.asarray(self.datasets.train.utterances[0]).shape)))
          model.add(Dense(PARAMS["batch_size"], activation='relu'))
          model.add(Dense(PARAMS["batch_size"], activation='relu'))
          model.add(Dense(np.asarray(self.datasets.train.intents[0]).shape[0], activation=PARAMS["activation"]))
          model.summary()

          tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=PARAMS["log_dir"], write_graph=True)

          model.compile(loss=PARAMS["loss"], optimizer=PARAMS["optimizer"], metrics=PARAMS["metrics"])
          monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=PARAMS["patience"], verbose=0, mode='auto')
          checkpointer = ModelCheckpoint(filepath=self.CONFIG["MODEL_WEIGHTS_PATH"], verbose=0, save_best_only=True) # Save best model
          model.fit(np.asarray(self.datasets.train.utterances), np.asarray(self.datasets.train.intents), epochs=PARAMS["epochs"], batch_size=PARAMS["batch_size"],  verbose=1, validation_split=0.05, callbacks=[tbCallBack, monitor, checkpointer])
          model.load_weights(self.CONFIG["MODEL_WEIGHTS_PATH"]) # load weights from best model
          scores = model.evaluate(np.asarray(self.datasets.train.utterances), np.asarray(self.datasets.train.intents))
          print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
          model.save(self.CONFIG["MODEL_PATH"])
          print("<<<<<<<< ML MODEL CREATED AND SAVED >>>>>>>>>>>\n\n")

    def load_keras_model(self):
        model = load_model(self.CONFIG["MODEL_PATH"])
        model.load_weights(self.CONFIG["MODEL_WEIGHTS_PATH"]) # load weights from best model
        return model

    def predict(self, text):
        ERROR_THRESHOLD = 0.25
        model = self.load_keras_model()
        toPredict = self.data_handler.convert_to_predict(text)
        if (toPredict.ndim == 1):
            toPredict = np.array([toPredict])

        results = model.predict(np.array(toPredict))[0]
        # filter out predictions below a threshold
        results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append((self.data_handler.intents[r[0]], r[1]))
        # return tuple of intent and probability
        return return_list
