
import sys
import numpy as np
import json
from data_handler import convert_to_predict

import urllib3, requests, json, base64, time, os, wget
from watson_machine_learning_client import WatsonMachineLearningAPIClient

with open('config.json', 'r') as f:
    global CONFIG
    CONFIG = json.load(f)

def get_scoring_url():
    wml_credentials=CONFIG["wml_credentials"]
    global client
    client = WatsonMachineLearningAPIClient(wml_credentials)
    # print(client.repository.list_models())
    # print(client.deployments.get_details())
    deployment_details = client.deployments.get_details(CONFIG["deployment_id"]);
    scoring_url = client.deployments.get_scoring_url(deployment_details)
    print(scoring_url)
    # scoring_url = 'https://ibm-watson-ml.mybluemix.net/v3/wml_instances/e7e44faf-ff8d-4183-9f37-434e2dcd6852/deployments/9ed7fe34-d927-4111-8b95-0e72a3bde6f8/online'
    return scoring_url

def get_results(sentence):
    ERROR_THRESHOLD = 0.25
    # print("Going to Classify: >> ", sentence)
    to_predict_arr, classes = convert_to_predict('data/data.csv', sentence)

    if (to_predict_arr.ndim == 1):
        to_predict_arr = np.array([to_predict_arr])

    scoring_data = {'values': to_predict_arr.tolist()}
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
        return_list.append((classes[r[0]], r[1]))
    return return_list

def classify():
    print("Model is ready! You now can enter requests.")
    for query in sys.stdin:
        print(get_results(query))

classify()
