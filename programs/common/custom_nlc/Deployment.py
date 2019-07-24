#!/usr/bin/env python

# python Deployment.py --data_dir data --result_dir results --config_file model_config.json --data_file data.csv


import tensorflow as tf
import argparse
import sys
import os
from os import environ
import zipfile
import types
import pandas as pd
from time import sleep

from build_code.handlers.data_handler import DataHandler
from build_code.handlers.cos_handler import COSHandler

import urllib3, requests, json, base64, time, os, wget

from watson_machine_learning_client import WatsonMachineLearningAPIClient

buckets = ['training-data-628e2c7c-5cd4-4b65-97d3-2d22b9972a73', 'training-results-628e2c7c-5cd4-4b65-97d3-2d22b9972a73']
scoring_params = None

with open('config.json', 'r') as f:
    global SECRET_CONFIG
    SECRET_CONFIG = json.load(f)

client = WatsonMachineLearningAPIClient(SECRET_CONFIG["wml_credentials"])
cos_handler = COSHandler(SECRET_CONFIG)

FLAGS = None

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def set_config():
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

    DATA_FILE_PATH = os.path.join(DATA_DIR, FLAGS.data_file)
    MODEL_PATH = os.path.join(RESULT_DIR, "model", MODEL_CONFIG["model_name"])
    MODEL_WEIGHTS_PATH = os.path.join(RESULT_DIR, "model", MODEL_CONFIG["model_weights"])
    if environ.get('JOB_STATE_DIR') is not None:
        LOG_DIR = os.path.join(os.environ["JOB_STATE_DIR"], MODEL_CONFIG["log_dir"])
    else:
        LOG_DIR = os.path.join(RESULT_DIR, MODEL_CONFIG["log_dir"])
    ensure_dir(DATA_FILE_PATH)
    ensure_dir(MODEL_PATH)
    global CONFIG
    CONFIG = {
                "DATA_DIR": DATA_DIR,
                "RESULT_DIR": RESULT_DIR,
                "DATA_FILE_PATH": DATA_FILE_PATH,
                "MODEL_PATH": MODEL_PATH,
                "MODEL_WEIGHTS_PATH": MODEL_WEIGHTS_PATH,
                "LOG_DIR": LOG_DIR,
                "MODEL_CONFIG": MODEL_CONFIG
             }

def train_model():
    model_definition_metadata = {
                client.repository.DefinitionMetaNames.NAME: "HomeAutomation_NLC_Model definition",
                client.repository.DefinitionMetaNames.DESCRIPTION: "HomeAutomation_ML_Model description",
                client.repository.DefinitionMetaNames.AUTHOR_NAME: "Gurvinder Singh",
                client.repository.DefinitionMetaNames.FRAMEWORK_NAME: "tensorflow",
                client.repository.DefinitionMetaNames.FRAMEWORK_VERSION: "1.5",
                client.repository.DefinitionMetaNames.RUNTIME_NAME: "python",
                client.repository.DefinitionMetaNames.RUNTIME_VERSION: "3.5",
                client.repository.DefinitionMetaNames.EXECUTION_COMMAND: "python3 build_code/IntentClassification.py --config_file model_config.json --data_file data.csv"
                }

    compressed_code = "build_code.zip"
    definition_details = client.repository.store_definition(compressed_code, model_definition_metadata)
    definition_uid = client.repository.get_definition_uid(definition_details)
    print("definition_uid: ", definition_uid)

    # Configure the training metadata for the TRAINING_DATA_REFERENCE and TRAINING_RESULTS_REFERENCE.
    training_configuration_metadata = {
                client.training.ConfigurationMetaNames.NAME: "HomeAutomation_NLC_Model",
                client.training.ConfigurationMetaNames.AUTHOR_NAME: "Gurvinder Singh",
                client.training.ConfigurationMetaNames.DESCRIPTION: "HomeAutomation_ML_Model training description",
                client.training.ConfigurationMetaNames.COMPUTE_CONFIGURATION: {"name": "k80"},
                client.training.ConfigurationMetaNames.TRAINING_DATA_REFERENCE: {
                    "connection": {
                        "endpoint_url": SECRET_CONFIG["service_endpoint"],
                        "access_key_id": SECRET_CONFIG["cos_credentials"]['cos_hmac_keys']['access_key_id'],
                        "secret_access_key": SECRET_CONFIG["cos_credentials"]['cos_hmac_keys']['secret_access_key']
                    },
                    "source": {
                        "bucket": buckets[0],
                    },
                    "type": "s3"
                },
                client.training.ConfigurationMetaNames.TRAINING_RESULTS_REFERENCE: {
                    "connection": {
                        "endpoint_url": SECRET_CONFIG["service_endpoint"],
                        "access_key_id": SECRET_CONFIG["cos_credentials"]['cos_hmac_keys']['access_key_id'],
                        "secret_access_key": SECRET_CONFIG["cos_credentials"]['cos_hmac_keys']['secret_access_key']
                    },
                    "target": {
                        "bucket": buckets[1],
                    },
                    "type": "s3"
                }
            }

    training_run_details = client.training.run(definition_uid, training_configuration_metadata)
    training_run_guid_async = client.training.get_run_uid(training_run_details)
    return training_run_guid_async

def store_model(trained_model_guid):
    print("IN store_model: >>> ", trained_model_guid)
    metadata = {
        client.repository.ModelMetaNames.NAME: 'HomeAutomation_NLC_Model',
        client.repository.ModelMetaNames.AUTHOR_NAME: 'Gurvinder Singh',
        client.repository.ModelMetaNames.FRAMEWORK_NAME: 'tensorflow',
        client.repository.ModelMetaNames.FRAMEWORK_VERSION: '1.5',
        client.repository.ModelMetaNames.RUNTIME_NAME: 'python',
        client.repository.ModelMetaNames.RUNTIME_VERSION: '3.5',
        client.repository.ModelMetaNames.FRAMEWORK_LIBRARIES: [{'name':'keras', 'version': '2.1.3'}]
        }
    saved_model_details = client.repository.store_model(trained_model_guid, meta_props=metadata)
    # filename = "results/my_nlc_model.h5"
    # tar_filename = filename + ".tgz"
    # cmdstring = "tar -zcvf " + tar_filename + " " + filename
    # os.system(cmdstring);
    # saved_model_details = client.repository.store_model(tar_filename, metadata)
    return saved_model_details

def deploy_model(model_uid):
    deployment_details = client.deployments.create(model_uid, "HomeAutomation_NLC_Model_Deploy")
    scoring_url = client.deployments.get_scoring_url(deployment_details)
    return scoring_url

def scoring_function(params=scoring_params):

    def score(payload):
        try:
            import re
            from watson_machine_learning_client import WatsonMachineLearningAPIClient
            client = WatsonMachineLearningAPIClient(params['wml_credentials'])

            maxlen = 50

            preprocessed_records = []
            complain_data = payload['values']
            word_index = params['word_index']

            for data in complain_data:
                comment = data[0]
                cleanString = re.sub(r"[!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~]", "", comment)
                splitted_comment = cleanString.split()[:maxlen]
                hashed_tokens = []

                for token in splitted_comment:
                    index = word_index.get(token, 0)
                    if index < 501 and index > 0:
                        hashed_tokens.append(index)

                hashed_tokens_size = len(hashed_tokens)
                padded_tokens = [0]*(maxlen-hashed_tokens_size) + hashed_tokens
                preprocessed_records.append(padded_tokens)

            scoring_payload = {'values': preprocessed_records}
            print(str(scoring_payload))
            return client.deployments.score(params['scoring_endpoint'], scoring_payload)

        except Exception as e:
            return { "error" : repr( e ) }

    return score

def retrain_model(definition_uid):
    training_configuration_metadata = {
                client.training.ConfigurationMetaNames.NAME: "HomeAutomation_ML_Model_Retrain1",
                client.training.ConfigurationMetaNames.AUTHOR_NAME: "Gurvinder Singh",
                client.training.ConfigurationMetaNames.DESCRIPTION: "HomeAutomation_ML_Model re-training description",
                client.training.ConfigurationMetaNames.COMPUTE_CONFIGURATION: {"name": "k80"},
                client.training.ConfigurationMetaNames.TRAINING_DATA_REFERENCE: {
                        "connection": {
                            "endpoint_url": SECRET_CONFIG["service_endpoint"],
                            "aws_access_key_id": cos_credentials['cos_hmac_keys']['access_key_id'],
                            "aws_secret_access_key": cos_credentials['cos_hmac_keys']['secret_access_key']
                        },
                        "source": {
                            "bucket": buckets[0],
                        },
                        "type": "s3"
                    },
                client.training.ConfigurationMetaNames.TRAINING_RESULTS_REFERENCE: {
                    "connection": {
                        "endpoint_url": SECRET_CONFIG["service_endpoint"],
                        "aws_access_key_id": cos_credentials['cos_hmac_keys']['access_key_id'],
                        "aws_secret_access_key": cos_credentials['cos_hmac_keys']['secret_access_key']
                    },
                    "target": {
                        "bucket": buckets[1],
                    },
                    "type": "s3"
                }
            }

    training_run_details = client.training.run(definition_uid, training_configuration_metadata)
    training_run_guid_async = client.training.get_run_uid(training_run_details)
    print("training_run_guid_async: >> ", training_run_guid_async)
    # Get training run status.
    status = client.training.get_status(training_run_guid_async)
    print(json.dumps(status, indent=2))
    client.training.monitor_logs(training_run_guid_async)

def update_model(model_uid):
    model_content = "HomeAutomation_NLC_Model.tar.gz"
    model_details = client.repository.update_model(model_uid, model_content)
    print(json.dumps(model_details, indent=2))

def update_deployment(deployment_uid):
    deployment_details = client.deployments.update(deployment_uid)
    print(json.dumps(deployment_details, indent=2))

def delete_model(artifact_uid):
    # artifact_uid = "9cd7108e-1310-4a72-8011-b59c16de268f"
    # client.repository.delete_model(artifact_uid)
    client.repository.delete(artifact_uid)

def get_model_details(model_uid):
    model_details = client.repository.get_model_details(model_uid)
    # with open('model_details.json', 'w') as outfile:
    #      json.dump(model_details, outfile)
    print(json.dumps(model_details, indent=2))

def details_to_file():
    details = client.repository.get_details()
    with open('details.json', 'w') as outfile:
         json.dump(details, outfile)

def delete_trainings(trainings):
    for t in trainings:
        client.training.delete(t)

def delete_all(trainings):
    # client.training.list()
    delete_trainings(trainings)
    details_to_file()
    # details = client.repository.get_details()
    with open('details.json') as f:
        data = json.load(f)
    for r in data["models"]["resources"]:
        client.repository.delete(r["metadata"]["guid"])
    for r in data["definitions"]["resources"]:
        client.repository.delete_definition(r["metadata"]["guid"])
    for r in data["runtimes"]["resources"]:
        client.runtimes.delete(r["metadata"]["guid"])
    for r in data["deployments"]["resources"]:
        client.deployments.delete(r["metadata"]["guid"])

def zipdir(path, ziph):
# ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

# This function does not work with the Lite plan
def deploy_scoring_function(scoring_endpoint):
    file = cos_handler.get_item(buckets[0], FLAGS.data_file)
    df = pd.read_csv(file["Body"])
    data_handler = DataHandler(df, "keras")
    scoring_params = {
        'scoring_endpoint': scoring_endpoint,
        'wml_credentials': SECRET_CONFIG["wml_credentials"],
        'word_index': data_handler.get_tokenizer().word_index,
        "intents": data_handler.get_intents()
    }
    # ai_function = scoring_function(scoring_params)
    def scoring_function(params=scoring_params):

        def score(payload):
            try:
                import re
                import numpy as np
                from watson_machine_learning_client import WatsonMachineLearningAPIClient
                client = WatsonMachineLearningAPIClient(params['wml_credentials'])

                preprocessed_records = []
                texts = payload['values']
                word_index = params['word_index']
                maxlen = 50
                for text in texts:
                    cleanString = re.sub(r"[!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~]", "", text)
                    splitted_text = cleanString.split()[:maxlen]
                    hashed_tokens = []
                    for token in splitted_text:
                        index = word_index.get(token, 0)
                        # index = scoring_params["word_index"].get(token, 0)
                        if index < 501 and index > 0:
                            hashed_tokens.append(index)

                    hashed_tokens_size = len(hashed_tokens)
                    padded_tokens = [0]*(maxlen - hashed_tokens_size) + hashed_tokens
                    preprocessed_records.append(padded_tokens)

                to_predict_arr = np.asarray(preprocessed_records)
                scoring_payload = {'values': to_predict_arr.tolist()}

                # print(str(scoring_payload))
                resp = client.deployments.score(params['scoring_endpoint'], scoring_payload)
                return_list = []
                ERROR_THRESHOLD = 0.15
                for val in resp["values"]:
                    result = val[0]
                    result = [[i,r] for i,r in enumerate(result) if r>ERROR_THRESHOLD]
                    # sort by strength of probability
                    result.sort(key=lambda x: x[1], reverse=True)
                    classifyResp = []
                    for r in result:
                        classifyResp.append((params["intents"][r[0]], r[1]))
                    return_list.append(classifyResp)
                return return_list

            except Exception as e:
                return { "error" : repr( e ) }

        return score

    # runtime_meta = {
    #                          client.runtimes.ConfigurationMetaNames.NAME: "Runtime specification",
    #                          client.runtimes.ConfigurationMetaNames.PLATFORM: {
    #                             "name": "python",
    #                             "version": "3.5"
    #                          }
    #                  }
    # runtime_details = client.runtimes.store(meta_props=runtime_meta)
    # print(runtime_details)
    # runtime_url = client.runtimes.get_url(runtime_details)
    # runtime_uid = "4ad69045-6280-4ba5-a80e-f8d31a6ea8cd"
    meta_data = {
        client.repository.FunctionMetaNames.NAME: 'MyNLC Scoring - AI Function'
        # client.repository.FunctionMetaNames.RUNTIME_UID: runtime_uid
    }

    function_details = client.repository.store_function(function=scoring_function, meta_props=meta_data)
    print("<<<<<< STORED FUNCTION_DETAILS: >>>>>>>> ")
    print(json.dumps(function_details, indent=2))
    function_uid = client.repository.get_function_uid(function_details)
    print("function_uid 1: ", function_uid);
    print("function_uid 2: ", function_details["metadata"]["guid"]);
    function_deployment_details = client.deployments.create(artifact_uid=function_details["metadata"]["guid"], name='MyNLC Scoring - AI Function Deployment')
    ai_function_scoring_endpoint = client.deployments.get_scoring_url(function_deployment_details)

    print(ai_function_scoring_endpoint)


def process_deployment():
    print("<<<<<< IN process_deployment >>>>>> ")
    zipf = zipfile.ZipFile('build_code.zip', 'w', zipfile.ZIP_DEFLATED)
    zipdir('build_code', zipf)
    zipf.close()
    training_run_guid_async = train_model()
    print(training_run_guid_async)
    # training_run_guid_async = 'training-ZknMig2ig'
    client.training.monitor_logs(training_run_guid_async)
    status = client.training.get_status(training_run_guid_async)
    while status["state"] != 'completed':
         print("Training Status:>> ", status["state"])
         sleep(5) # Sleep for 5 seconds
         status = client.training.get_status(training_run_guid_async)
    print("<<< Training Completed >>> ")

    saved_model_details = store_model(training_run_guid_async)
    print("Trained Model Guid: >> ", saved_model_details["metadata"]["guid"])
    scoring_endpoint = deploy_model(saved_model_details["metadata"]["guid"])
    # get_model_details("9d22bc1b-b0d1-4f76-9a3b-e9117387314f")
    # scoring_endpoint = deploy_model('4d449972-6e31-408d-bc24-4e9e64fd0584')
    print("scoring_endpoint", scoring_endpoint)
    deploy_scoring_function(scoring_endpoint)

def test_cos():
    files = cos_handler.get_bucket_contents(buckets[1])
    for file in files:
        print("Item: {0} ({1} bytes).".format(file.key, file.size))
    file = cos_handler.get_item(buckets[0], FLAGS.data_file)
    df = pd.read_csv(file["Body"])
    print(df)

def set_word_index():
    file = cos_handler.get_item(buckets[0], FLAGS.data_file)
    df = pd.read_csv(file["Body"])
    data_handler = DataHandler(df, "keras")
    cos_handler.create_file(buckets[0], 'word_index.json', json.dumps(data_handler.get_tokenizer().word_index, indent=2))

def main():
    set_config()
    # set_word_index()
    details_to_file()
    # trainings = client.training.list()
    # print(trainings)
    # delete_all(["model-wm2r9qjr"])
    # process_deployment()
    # saved_model_details = store_model('model-7omxjxd6')
    # print(type(saved_model_details))
    # print(json.dumps(saved_model_details, indent=2))
    # print("Model Guid: >> ", saved_model_details["metadata"]["guid"])
    # model = client.repository.load('a90e007d-663c-405b-96ae-9f3ff88993e6')
    # get_model_details("a90e007d-663c-405b-96ae-9f3ff88993e6")
    # scoring_endpoint = deploy_model('a90e007d-663c-405b-96ae-9f3ff88993e6')
    # print("scoring_endpoint", scoring_endpoint)
    # scoring_endpoint = "https://ibm-watson-ml.mybluemix.net/v3/wml_instances/e7e44faf-ff8d-4183-9f37-434e2dcd6852/deployments/55717edb-e642-4318-b4e6-f917ca6d6bc1/online"
    # deploy_scoring_function(scoring_endpoint)
    # delete_model("75e3cac1-d456-4f4c-8342-710ccf9fb287") # provide artifact_uid
    # list_modals = client.repository.list_models()
    # print(list_modals)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # environment variable when name starts with $
  parser.add_argument('--data_dir', type=str, default='$DATA_DIR', help='Directory with data')
  parser.add_argument('--result_dir', type=str, default='$RESULT_DIR', help='Directory with results')
  parser.add_argument('--data_file', type=str, default='data.csv', help='File name for Intents and Classes')
  parser.add_argument('--config_file', type=str, default='model_config.json', help='Model Configuration file name')
  parser.add_argument('--from_cloud', type=int, default=True, help='Deploy using data file from COS')

  FLAGS, unparsed = parser.parse_known_args()
  print("Start Process.....")
  main()
  # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

# client.repository.download("dfd5b3c3-e95e-43c2-8f2f-8010d87a221a", 'HomeAutomation_ML_Model.tar.gz')

# details_to_file() # fetches all details to a json file
# delete_trainings()
# delete_model("75e3cac1-d456-4f4c-8342-710ccf9fb287") # provide artifact_uid
# delete_all()

# train_model()
# client.training.list()
# definition_details = client.repository.get_definition_details()
# list_modals = client.repository.list_models()
# print(list_modals)

# store_model("training-fG3KcL5mg") # provide run_guid
# deploy_model("78c36c3f-b80e-4643-b642-c43b6720c25f") # provide model_uid
# get_model_details("78c36c3f-b80e-4643-b642-c43b6720c25f")

# retrain_model("1162d2a2-f0fb-4293-9e25-2327e7017e79") # Provide definition_uid
# client.repository.list_models()
# update_model("00c8653e-3022-4942-934f-c53dd287eec1")
# client.deployments.list()
# update_deployment("ed7c91b1-79b5-4e2a-bc4e-1fc5bf16f1f0")
# update_deployment("73b5460a-7cf1-4be3-be42-fd8353f1de45")
