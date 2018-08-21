#!/usr/bin/env python

#  Author: Gurvinder Singh
#  Date: 01/08/2018
#
# Natural Language Classification.
#
# python IntentClassification.py --data_dir ../../../data --data_file data.csv --result_dir results
#
# *************************************** #

"""A very simple NLC classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import tarfile

from handlers.data_handler import DataHandler
# from handlers.scikit_model_handler import ModelHandler
from handlers.keras_model_handler import ModelHandler
import pandas as pd
import numpy as np
import random

import tensorflow as tf

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

def get_keras_model(data_handler):
    print("\n\n <<<<<<<< GET KERAS MODEL HANDLER >>>>>>>>")
    # Initialize a Random Forest classifier with 100 trees
    CONFIG = {
                "MODEL_PATH": MODEL_PATH,
                "MODEL_WEIGHTS_PATH": MODEL_WEIGHTS_PATH
             }

    model_handler = ModelHandler(data_handler, CONFIG)
    return model_handler

def get_scikit_model(data_handler):
    print("\n\n <<<<<<<< GET SCIKIT MODEL HANDLER >>>>>>>>")
    # Initialize a Random Forest classifier with 100 trees
    CONFIG = {"MODEL_PATH": MODEL_PATH}
    model_handler = ModelHandler(data_handler, CONFIG)
    return model_handler

def get_model_handler(library_name="keras"):
    global df
    global dh
    # df = pd.read_csv('../../../data/raw_home_automation.csv', header=0, delimiter=",")
    df = pd.read_csv(DATA_FILE_PATH, header=0, delimiter=",")
    dh = DataHandler(df, library_name)
    if library_name == "scikit":
        return get_scikit_model(dh)
    elif library_name == "keras":
        return get_keras_model(dh)
    else:
        return None

def create_model(library_name="keras"):
    model_handler = get_model_handler(library_name)
    print(model_handler.name)
    if library_name == "scikit":
        print("\n\n <<<<<<<< CREATE MODEL FROM SCIKIT LIBRARY >>>>>>>>")
        PARAMS = {"n_estimators": 100}
        model_handler.create_model(PARAMS)
    elif library_name == "keras":
        print("\n\n <<<<<<<< CREATE MODEL FROM KERAS LIBRARY >>>>>>>>")
        PARAMS = {
                    "log_dir": TENSORBOARD_LOGS_PATH,
                    "epochs": FLAGS.epochs,
                    "batch_size": FLAGS.batch_size,
                    "activation": FLAGS.activation,
                    "loss": FLAGS.loss,
                    "optimizer": FLAGS.optimizer,
                    "metrics": ["accuracy"],
                    "patience": 10
                 }
        model_handler.create_model(PARAMS)
    else:
        return None

def main(_):
    set_flags()
    create_model("keras")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # environment variable when name starts with $
  parser.add_argument('--data_dir', type=str, default='$DATA_DIR', help='Directory with data')
  parser.add_argument('--result_dir', type=str, default='$RESULT_DIR', help='Directory with results')
  parser.add_argument('--data_file', type=str, default='data.csv', help='File name for Intents and Classes')
  parser.add_argument('--loss', type=str, default='binary_crossentropy', help='loss function: categorical_crossentropy, mean_squared_error')
  parser.add_argument('--activation', type=str, default='softmax', help='activation function: softmax, sigmoid')
  parser.add_argument('--optimizer', type=str, default='adam', help='optimizer : adam, rmsprop')
  parser.add_argument('--epochs', type=int, default=200, help='Number of training iterations')
  parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
  parser.add_argument('--model_name', type=str, default='my_nlc_model.h5', help='Name of the model')

  FLAGS, unparsed = parser.parse_known_args()
  print("Start model training")
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
