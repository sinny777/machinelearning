
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

RESULTS = {}

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
    RESULTS["model_uid"] = model_uid
    deployment_details = client.deployments.create(model_uid, FLAGS.model_name)
    scoring_url = client.deployments.get_scoring_url(deployment_details)
    RESULTS["scoring_url"] = scoring_url
    return scoring_url

def main():
    print("Lets Deploy your model...\n\n")
    # init_steps()
    #FLAGS.username = getpass.getuser()
    FLAGS.model_name = "model_"+FLAGS.username
    # print("Your model_name: >> ", FLAGS.model_name)
    RESULTS["username"] = FLAGS.username
    scoring_url = deploy_model()
    # scoring_url = 'https://us-south.ml.cloud.ibm.com/v3/wml_instances/5b0ab00a-3bc8-4c6b-9ba2-c6362e633958/deployments/22f3814e-fd4b-4deb-8a22-f091ee694e3d/online'
    # RESULTS["model_uid"] = '2d5145d0-5743-436c-be2f-71e6f5bfe119'
    RESULTS["scoring_url"] = scoring_url
    print("RESULTS: >>> ", RESULTS)
    print("\n\nwml_credentials: >> ", wml_credentials)
    # submit_details()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_path', type=str, help='Saved Model Path')
  parser.add_argument('--framework_name', default="scikit-learn", type=str, help='scikit-learn, keras etc.')
  parser.add_argument('--username', type=str, help='username')

  FLAGS, unparsed = parser.parse_known_args()
  main()
