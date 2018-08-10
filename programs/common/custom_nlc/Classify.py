
## python Classify.py --data_dir ../../../data --data_file data.csv --result_dir results

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
import numpy as np
import json

import urllib3, requests, json, base64, time, os, wget
from watson_machine_learning_client import WatsonMachineLearningAPIClient

# from handlers.scikit_model_handler import ModelHandler
from handlers.keras_model_handler import ModelHandler
from handlers.data_handler import DataHandler

FLAGS = None
library_name = "keras"

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

with open('config.json', 'r') as f:
    global CONFIG
    CONFIG = json.load(f)

def get_keras_model(data_handler):
    # Initialize a Random Forest classifier with 100 trees
    CONFIG = {
                "MODEL_PATH": MODEL_PATH,
                "MODEL_WEIGHTS_PATH": MODEL_WEIGHTS_PATH
             }

    model_handler = ModelHandler(data_handler, CONFIG)
    return model_handler

def get_model_handler(library_name="keras"):
    global df
    global dh
    # df = pd.read_csv('../../../data/raw_home_automation.csv', header=0, delimiter=",")
    df = pd.read_csv(DATA_FILE_PATH, header=0, delimiter=",")
    dh = DataHandler(df, library_name)
    if library_name == "scikit":
        print("\n\n <<<<<<<< GET SCIKIT MODEL HANDLER >>>>>>>>")
        return get_scikit_model(dh)
    elif library_name == "keras":
        print("\n\n <<<<<<<< GET KERAS MODEL HANDLER >>>>>>>>")
        return get_keras_model(dh)
    else:
        return None

def get_scoring_url():
    wml_credentials=CONFIG["wml_credentials"]
    global client
    client = WatsonMachineLearningAPIClient(wml_credentials)
    deployment_details = client.deployments.get_details(CONFIG["deployment_id"]);
    scoring_url = client.deployments.get_scoring_url(deployment_details)
    print("scoring_url: >> ", scoring_url)
    # scoring_url = 'https://ibm-watson-ml.mybluemix.net/v3/wml_instances/e7e44faf-ff8d-4183-9f37-434e2dcd6852/deployments/9ed7fe34-d927-4111-8b95-0e72a3bde6f8/online'
    return scoring_url

def get_results(sentence):
    ERROR_THRESHOLD = 0.25
    # print("Going to Classify: >> ", sentence)
    global model_handler
    global scoring_url
    try:
      model_handler
    except NameError:
      model_handler = get_model_handler(library_name)

    if FLAGS.from_cloud:
        ERROR_THRESHOLD = 0.25
        to_predict_arr = model_handler.data_handler.convert_to_predict(sentence)
        if (to_predict_arr.ndim == 1):
            to_predict_arr = np.array([to_predict_arr])

        scoring_data = {'values': to_predict_arr.tolist()}
        try:
          scoring_url
        except NameError:
          scoring_url = get_scoring_url()
        # send scoring dictionary to deployed model to get predictions
        resp = client.deployments.score(scoring_url, scoring_data)
        result = resp["values"][0][0]
        # filter out predictions below a threshold
        result = [[i,r] for i,r in enumerate(result) if r>ERROR_THRESHOLD]
        # sort by strength of probability
        result.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in result:
            return_list.append((model_handler.data_handler.intents[r[0]], r[1]))
        return return_list
    else:
        return model_handler.predict(sentence)



def classify(_):
    set_flags()
    print("Model is ready! You now can enter requests.")
    for query in sys.stdin:
        if query.strip() == "close":
            sys.exit(0)
        print(get_results(query.strip()))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # environment variable when name starts with $
  parser.add_argument('--data_dir', type=str, default='$DATA_DIR', help='Directory with data')
  parser.add_argument('--result_dir', type=str, default='$RESULT_DIR', help='Directory with results')
  parser.add_argument('--data_file', type=str, default='data.csv', help='File name for Intents and Classes')
  parser.add_argument('--model_name', type=str, default='my_nlc_model.h5', help='Name of the model')
  parser.add_argument('--from_cloud', type=int, default=True, help='Classify from Model deployed on IBM Cloud')

  FLAGS, unparsed = parser.parse_known_args()
  print("Start model training")
  tf.app.run(main=classify, argv=[sys.argv[0]] + unparsed)
