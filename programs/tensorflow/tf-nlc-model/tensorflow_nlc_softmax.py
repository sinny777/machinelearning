# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple NLC classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import sys
import os
import tarfile

from data_handler import read_data_sets, convert_to_predict

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input, concatenate, Activation
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
# from tf.keras.layers.pooling import GlobalMaxPooling1D, MaxPooling1D
# from tf.keras.layers.core import Dropout
# from keras import backend as K
# os.environ['KERAS_BACKEND']='theano'

FLAGS = None

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def set_flags():
    # print(FLAGS)
    global DATA_FILE_PATH
    global MODEL_PATH
    global MODEL_WEIGHTS_PATH
    global TENSORBOARD_LOGS_PATH
    if (FLAGS.data_dir[0] == '$'):
      DATA_DIR = os.environ[FLAGS.data_dir[1:]]
    else:
      DATA_DIR = FLAGS.data_dir
    if (FLAGS.result_dir[0] == '$'):
      RESULT_DIR = os.environ[FLAGS.result_dir[1:]]
    else:
      RESULT_DIR = FLAGS.result_dir

    DATA_FILE_PATH = os.path.join(DATA_DIR, FLAGS.data_file)
    MODEL_PATH = os.path.join(RESULT_DIR, FLAGS.model_name)
    MODEL_WEIGHTS_PATH = os.path.join(RESULT_DIR, "model_weights.hdf5")
    TENSORBOARD_LOGS_PATH = os.path.join(RESULT_DIR, "tensorboard_logs")
    ensure_dir(DATA_FILE_PATH)
    ensure_dir(MODEL_PATH)

def get_results(sentence):
    ERROR_THRESHOLD = 0.25
    set_flags()
    to_predict_arr, classes = convert_to_predict(DATA_FILE_PATH, sentence)
    print("to_predict: >>> ")
    print(to_predict_arr)
    if (to_predict_arr.ndim == 1):
        to_predict_arr = np.array([to_predict_arr])

    model = load_model(MODEL_PATH)
    results = model.predict([to_predict_arr])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

def classify(_):
    print("Model is ready! You now can enter requests.")
    for query in sys.stdin:
        print(get_results(query))

def main(_):
  set_flags()
  # Import data
  nlc_data = read_data_sets(DATA_FILE_PATH)

  init_g = tf.global_variables_initializer()
  init_l = tf.local_variables_initializer()
  with tf.Session() as sess:
    sess.run(init_g)
    sess.run(init_l)
    model = Sequential()
    # model.add(Dense(output_dim=8,init ='uniform',activation='relu', input_dim=len(train_x[0])))
    model.add(Dense(8, activation='relu', input_shape=(np.asarray(nlc_data.train.intents[0]).shape)))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(np.asarray(nlc_data.train.classes[0]).shape[0], activation='softmax'))
    model.summary()

    tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_LOGS_PATH, write_graph=True)

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, verbose=0, mode='auto')
    checkpointer = ModelCheckpoint(filepath=MODEL_WEIGHTS_PATH, verbose=0, save_best_only=True) # Save best model
    model.fit(np.asarray(nlc_data.train.intents), np.asarray(nlc_data.train.classes), epochs=FLAGS.training_iters, batch_size=FLAGS.batch_size,  verbose=1, validation_split=0.1, callbacks=[tbCallBack, monitor, checkpointer])
    model.load_weights(MODEL_WEIGHTS_PATH) # load weights from best model
    scores = model.evaluate(np.asarray(nlc_data.train.intents), np.asarray(nlc_data.train.classes))
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    model.save(MODEL_PATH)
    print("<<<<<<<< ML MODEL CREATED AND SAVED >>>>>>>>>>>\n\n")
    # classify()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # environment variable when name starts with $
  parser.add_argument('--data_dir', type=str, default='$DATA_DIR', help='Directory with data')
  parser.add_argument('--result_dir', type=str, default='$RESULT_DIR', help='Directory with results')
  parser.add_argument('--data_file', type=str, default='data.csv', help='File name for Intents and Classes')
  parser.add_argument('--training_iters', type=int, default=200, help='Number of training iterations')
  parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
  parser.add_argument('--model_name', type=str, default='my_nlc_model.h5', help='Name of the model')

  FLAGS, unparsed = parser.parse_known_args()
  print("Start model training")
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
