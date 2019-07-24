
# python publish.py --model_path "gurvsin3_model.pkl" --framework_name "scikit-learn"

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
from os import environ
import tarfile
import pickle
from sklearn.externals import joblib

import pandas as pd
import numpy as np
import random
import re

import numpy as np
import json
import os
import getpass

# import urllib3, requests, json, base64, time, os, wget
from watson_machine_learning_client import WatsonMachineLearningAPIClient

# from repository.mlrepository import MetaNames
# from repository.mlrepository import MetaProps
# from repository.mlrepositoryclient import MLRepositoryClient
# from repository.mlrepositoryartifact import MLRepositoryArtifact

RESULT = {}

FLAGS = None

wml_credentials = {
                  "apikey": "bzD4Q2pxSX9Ug15pYGpwtQopmvletsJ5Dpkks5cscaWO",
                  "iam_apikey_description": "Auto generated apikey during resource-key operation for Instance - crn:v1:bluemix:public:pm-20:us-south:a/7081a1b17b89ae7f7c9b01f40e0be431:5b0ab00a-3bc8-4c6b-9ba2-c6362e633958::",
                  "iam_apikey_name": "auto-generated-apikey-dd250624-9978-4dfa-9a55-8107eca3bc24",
                  "iam_role_crn": "crn:v1:bluemix:public:iam::::serviceRole:Writer",
                  "iam_serviceid_crn": "crn:v1:bluemix:public:iam-identity::a/7081a1b17b89ae7f7c9b01f40e0be431::serviceid:ServiceId-3d715295-c209-4c04-ac02-943d9fd6cbfd",
                  "instance_id": "5b0ab00a-3bc8-4c6b-9ba2-c6362e633958",
                  "password": "2f068d39-881b-421a-b185-6246193aaee2",
                  "url": "https://us-south.ml.cloud.ibm.com",
                  "username": "dd250624-9978-4dfa-9a55-8107eca3bc24"
                }

client = WatsonMachineLearningAPIClient(wml_credentials)


# base_path = "https://ibm-watson-ml.mybluemix.net"
# username = "admin"
# password = "goldbug2"
#
# ml_repository_client = MLRepositoryClient(base_path)
# ml_repository_client.authorize(username, password)

def deploy_model():
    model = joblib.load(FLAGS.model_path)
    metadata = {
        client.repository.ModelMetaNames.AUTHOR_NAME: FLAGS.username,
        client.repository.ModelMetaNames.NAME: FLAGS.model_name,
        client.repository.ModelMetaNames.RUNTIME_NAME: 'python',
        client.repository.ModelMetaNames.RUNTIME_VERSION: '3.5',
        client.repository.ModelMetaNames.FRAMEWORK_NAME: FLAGS.framework_name
        }
    saved_model_details = client.repository.store_model(model, metadata)
    print(saved_model_details)
    model_uid = saved_model_details["metadata"]["guid"]
    RESULT["model_uid"] = model_uid
    deployment_details = client.deployments.create(model_uid, FLAGS.model_name)
    scoring_url = client.deployments.get_scoring_url(deployment_details)
    RESULT["scoring_url"] = scoring_url
    return scoring_url

def get_test_data():
    df_data_1 = pd.read_csv(os.environ['DSX_PROJECT_DIR']+'/datasets/'+FLAGS.test_file_path, delimiter=",")
    df_train = pd.get_dummies(df_data_1, columns=["Pclass","Embarked","Sex"])
    df_train.fillna(df_train["Age"].median(skipna=True), inplace=True)
    df_train.drop('Sex_female', axis=1, inplace=True)
    df_train.drop('PassengerId', axis=1, inplace=True)
    df_train.drop('Name', axis=1, inplace=True)
    df_train.drop('Ticket', axis=1, inplace=True)
    df_train.drop('Cabin', axis=1, inplace=True)
    df_train.head()

    X = df_train[["Age","SibSp","Parch","Fare","Pclass_1","Pclass_2","Pclass_3","Embarked_C","Embarked_Q","Embarked_S","Sex_male"]]
    return X

def test_model_local():
    X = get_test_data()
    return_list = []
    saved_model = joblib.load(FLAGS.model_path)
    resp = saved_model.predict(X)
    df = pd.read_csv(os.environ['DSX_PROJECT_DIR']+'/datasets/'+FLAGS.test_file_path, delimiter=",")
    df = df.join(pd.DataFrame({'prediction': resp}))
    print(df.head())
    df.to_csv(os.environ['DSX_PROJECT_DIR']+'/datasets/'+FLAGS.username+'_test_results.csv', index=False)

def test_model_cloud(scoring_url):
    print(scoring_url)
    X = get_test_data()
    scoring_data = {'values': X.values.tolist()}
    resp = client.deployments.score(scoring_url, scoring_data)
    print(len(resp["values"]))
    print(resp["values"][0][0])
    result = []
    for i in range(0, len(resp["values"])):
        result.append(resp["values"][i][0])

    df = pd.read_csv(os.environ['DSX_PROJECT_DIR']+'/datasets/'+FLAGS.test_file_path, delimiter=",")
    df = df.join(pd.DataFrame({'prediction': result}))
    print(df.head())
    df.to_csv(os.environ['DSX_PROJECT_DIR']+'/datasets/'+FLAGS.username+'_test_results.csv', index=False)

def check_accuracy():
    user_df = pd.read_csv(os.environ['DSX_PROJECT_DIR']+'/datasets/gurvsin_test_results.csv', header=0, delimiter=",")
    main_df = pd.read_csv(os.environ['DSX_PROJECT_DIR']+'/datasets/Titanictrain -- Blinddataset -- WithAnswers.csv', header=0, delimiter=",")
    matches_found = 0
    miss_matches = 0
    for i in range(0, len(main_df)):
        print(main_df.iloc[i,-1]," : ", user_df.iloc[i,-1])

        if main_df.iloc[i, -1] == user_df.iloc[i,-1]:
            matches_found = matches_found + 1
        else:
            miss_matches = miss_matches + 1

    print("matches_found: >> ", matches_found)
    print("miss_matched: >> ", miss_matches)
    print("accuracy: >>> ", matches_found/len(main_df) * 100)

def main():
    # print("Lets Publish the model...")
    # print("Model: >> ", type(FLAGS.model))
    # FLAGS.username = getpass.getuser()
    # print("FLAGS.username: >> ", FLAGS.username)
    # FLAGS.model_name = "model_"+FLAGS.username
    # print("FLAGS.model_name: >> ", FLAGS.model_name)
    # test_model_local()
    # scoring_url = deploy_model()
    # scoring_url = 'https://us-south.ml.cloud.ibm.com/v3/wml_instances/5b0ab00a-3bc8-4c6b-9ba2-c6362e633958/deployments/f50af8b8-6ddc-4684-b70f-55d8522006d6/online'
    # test_model(scoring_url)
    # test_model_local()
    # test_model_local()
    # test_model_cloud(scoring_url)
    check_accuracy()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_path', type=str, help='Saved Model Path')
  parser.add_argument('--framework_name', type=str, help='scikit-learn, keras etc.')
  parser.add_argument('--test_file_path', default='Titanictrain -- Blind.csv', type=str, help='Test File Path')

  FLAGS, unparsed = parser.parse_known_args()
  main()
