#!/usr/bin/env python

#  Author: Gurvinder Singh
#  Date: 01/08/2018
#
# Natural Language Classification.
#
# python build_code/ImageClassification.py --data_dir data/car-damage-dataset/data3a --result_dir results --config_file model_config.json
# python build_code/ImageClassification.py --data_dir data/cctv_detection --result_dir results --config_file model_config.json
#
# *************************************** #

"""A very simple NLC classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import matplotlib.pyplot as plt

import argparse
import sys
import os
from os import environ
import tarfile
import json

# import pandas as pd
# import numpy as np
# import random

# from handlers.scikit_model_handler import ModelHandler
# from handlers.keras_model_handler import ModelHandler

FLAGS = None

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def set_config():
    # print(FLAGS)
    if (FLAGS.data_dir[0] == '$'):
      DATA_DIR = os.environ[FLAGS.data_dir[1:]]
    else:
      DATA_DIR = FLAGS.data_dir
    if (FLAGS.result_dir[0] == '$'):
      RESULT_DIR = os.environ[FLAGS.result_dir[1:]]
    else:
      RESULT_DIR = FLAGS.result_dir

    with open(os.path.join(DATA_DIR, FLAGS.config_file), 'r') as f:
        MODEL_CONFIG = json.load(f)

    # DATA_FILE_PATH = os.path.join(DATA_DIR, FLAGS.data_file)
    MODEL_PATH = os.path.join(RESULT_DIR, "model", MODEL_CONFIG["model_name"])
    CHECKPOINTS_PATH = os.path.join(RESULT_DIR, "checkpoints/", MODEL_CONFIG["checkpoints_dir"])
    if environ.get('JOB_STATE_DIR') is not None:
        LOG_DIR = os.path.join(os.environ["JOB_STATE_DIR"], MODEL_CONFIG["log_dir"])
    else:
        LOG_DIR = os.path.join(RESULT_DIR, MODEL_CONFIG["log_dir"])
    # ensure_dir(DATA_FILE_PATH)
    ensure_dir(MODEL_PATH)
    global CONFIG
    CONFIG = {
                "DATA_DIR": DATA_DIR,
                "RESULT_DIR": RESULT_DIR,
                "MODEL_PATH": MODEL_PATH,
                "LOG_DIR": LOG_DIR,
                "MODEL_CONFIG": MODEL_CONFIG,
                "CHECKPOINTS_PATH": CHECKPOINTS_PATH
             }
    print(CONFIG)

def get_tf2_handler():
    print("\n\n <<<<<<<< GET TENSORFLOW 2.0 MODEL HANDLER >>>>>>>>")
    from handlers.tf2_model_handler import ModelHandler
    model_handler = ModelHandler(CONFIG)
    return model_handler

def get_model_handler():
    if FLAGS.framework == "tensorflow":
        return get_tf2_handler()   
    else:
        return None

def create_model(model_handler):
    print(model_handler.name)
    if FLAGS.framework == "tensorflow":
        print("\n\n <<<<<<<< CREATE MODEL FROM TENSORFLOW LIBRARY >>>>>>>>")
        return model_handler.create_model()
    else:
        return None

def main():
    set_config()
    model_handler = get_model_handler()

    classify = True
    
    if classify:
        image_path = os.path.join(CONFIG['DATA_DIR'], 'validation/accident/test10_22.jpg')
        # image_path = 'https://cdn1.sph.harvard.edu/wp-content/uploads/sites/30/2018/08/bananas-1354785_1920-1200x800.jpg'
        model_handler.predict_image(image_path)
    else:
        model = create_model(model_handler)
        if model != None:
            model, history = model_handler.train_model(model)
            model_handler.check_accuracy(history)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # environment variable when name starts with $
  parser.add_argument('--data_dir', type=str, default='$DATA_DIR', help='Directory with data')
  parser.add_argument('--result_dir', type=str, default='$RESULT_DIR', help='Directory with results')
  parser.add_argument('--config_file', type=str, default='model_config.json', help='Model Configuration file name')
  parser.add_argument('--framework', type=str, default='tensorflow', help='ML Framework to use')

  FLAGS, unparsed = parser.parse_known_args()
  print("Start model training")
#   tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  main()  
