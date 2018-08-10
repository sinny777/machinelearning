
# Install the boto library.
import ibm_boto3
from ibm_botocore.client import Config

import urllib3, requests, json, base64, time, os, wget

from watson_machine_learning_client import WatsonMachineLearningAPIClient

buckets = ['training-data-628e2c7c-5cd4-4b65-97d3-2d22b9972a73', 'training-results-628e2c7c-5cd4-4b65-97d3-2d22b9972a73']

with open('config.json', 'r') as f:
    global CONFIG
    CONFIG = json.load(f)

cos_credentials = CONFIG["cos_credentials"]

cos = ibm_boto3.resource('s3',
                         ibm_api_key_id=cos_credentials['apikey'],
                         ibm_service_instance_id=cos_credentials['resource_instance_id'],
                         ibm_auth_endpoint=CONFIG["auth_endpoint"],
                         config=Config(signature_version='oauth'),
                         endpoint_url=CONFIG["service_endpoint"])

client = WatsonMachineLearningAPIClient(CONFIG["wml_credentials"])

# print(list(cos.buckets.all()))
def show_bucket_files():
    for bucket_name in buckets:
        print(bucket_name)
        bucket_obj = cos.Bucket(bucket_name)
        for obj in bucket_obj.objects.all():
            print("  File: {}, {:4.2f}kB".format(obj.key, obj.size/1024))

def train_model():
    model_definition_metadata = {
                client.repository.DefinitionMetaNames.NAME: "MyCarNLC_ML_Model definition",
                client.repository.DefinitionMetaNames.DESCRIPTION: "MyCarNLC_ML_Model description",
                client.repository.DefinitionMetaNames.AUTHOR_NAME: "Gurvinder Singh",
                client.repository.DefinitionMetaNames.FRAMEWORK_NAME: "tensorflow",
                client.repository.DefinitionMetaNames.FRAMEWORK_VERSION: "1.5",
                client.repository.DefinitionMetaNames.RUNTIME_NAME: "python",
                client.repository.DefinitionMetaNames.RUNTIME_VERSION: "3.5",
                client.repository.DefinitionMetaNames.EXECUTION_COMMAND: "python3 custom_nlc/IntentClassification.py --epochs 200"
                }

    compressed_code = "../../../zip_files/custom_nlc.zip"
    definition_details = client.repository.store_definition(compressed_code, model_definition_metadata)
    definition_uid = client.repository.get_definition_uid(definition_details)
    print("definition_uid: ", definition_uid)

    # Configure the training metadata for the TRAINING_DATA_REFERENCE and TRAINING_RESULTS_REFERENCE.
    training_configuration_metadata = {
                client.training.ConfigurationMetaNames.NAME: "MyCarNLC_ML_Model",
                client.training.ConfigurationMetaNames.AUTHOR_NAME: "Gurvinder Singh",
                client.training.ConfigurationMetaNames.DESCRIPTION: "MyCarNLC_ML_Model training description",
                client.training.ConfigurationMetaNames.COMPUTE_CONFIGURATION: {"name": "k80"},
                client.training.ConfigurationMetaNames.TRAINING_DATA_REFERENCE: {
                        "connection": {
                            "endpoint_url": CONFIG["service_endpoint"],
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
                        "endpoint_url": CONFIG["service_endpoint"],
                        "aws_access_key_id": cos_credentials['cos_hmac_keys']['access_key_id'],
                        "aws_secret_access_key": cos_credentials['cos_hmac_keys']['secret_access_key']
                    },
                    "target": {
                        "bucket": buckets[1],
                    },
                    "type": "s3"
                },
            }

    training_run_details = client.training.run(definition_uid, training_configuration_metadata)
    training_run_guid_async = client.training.get_run_uid(training_run_details)
    # Get training run status.
    status = client.training.get_status(training_run_guid_async)
    print(json.dumps(status, indent=2))
    client.training.monitor_logs(training_run_guid_async)

def store_model(run_guid):
    meta_props = {"name": "MyCarNLC_ML_Model", "frameworkName": "tensorflow"}
    # model_path = "results/my_nlc_model.h5"
    saved_model_details = client.repository.store_model(run_guid, meta_props)
    print(json.dumps(saved_model_details, indent=2))

def deploy_model(model_uid):
    deployment_details = client.deployments.create(model_uid, "MyCarNLC_ML_Model_Deploy")
    scoring_url = client.deployments.get_scoring_url(deployment_details)
    print(scoring_url)

def retrain_model(definition_uid):
    training_configuration_metadata = {
                client.training.ConfigurationMetaNames.NAME: "MyCarNLC_ML_Model_Retrain1",
                client.training.ConfigurationMetaNames.AUTHOR_NAME: "Gurvinder Singh",
                client.training.ConfigurationMetaNames.DESCRIPTION: "MyCarNLC_ML_Model re-training description",
                client.training.ConfigurationMetaNames.COMPUTE_CONFIGURATION: {"name": "k80"},
                client.training.ConfigurationMetaNames.TRAINING_DATA_REFERENCE: {
                        "connection": {
                            "endpoint_url": CONFIG["service_endpoint"],
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
                        "endpoint_url": CONFIG["service_endpoint"],
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
    model_content = "MyCarNLC_ML_Model.tar.gz"
    model_details = client.repository.update_model(model_uid, model_content)
    print(json.dumps(model_details, indent=2))

def update_deployment(deployment_uid):
    deployment_details = client.deployments.update(deployment_uid)
    print(json.dumps(deployment_details, indent=2))

def delete_model(artifact_uid):
    # artifact_uid = "9cd7108e-1310-4a72-8011-b59c16de268f"
    client.repository.delete_model(artifact_uid)

def get_model_details(model_uid):
    model_details = client.repository.get_model_details(model_uid)
    print(json.dumps(model_details, indent=2))

def delete_trainings():
    trainings = ["training-SNaUKfcig"]
    for t in trainings:
        client.training.delete(t)

def details_to_file():
    details = client.repository.get_details()
    with open('details.json', 'w') as outfile:
         json.dump(details, outfile)

def delete_all():
    # details = client.repository.get_details()
    with open('details.json') as f:
        data = json.load(f)
    for r in data["definitions"]["resources"]:
        client.repository.delete_definition(r["metadata"]["guid"])


# client.repository.download("9e39aec2-3385-4b20-bda5-fe41f7dc4e9f", 'MyCarNLC_ML_Model.tar.gz')

details_to_file() # fetches all details to a json file
# delete_trainings()
# delete_model("9cd7108e-1310-4a72-8011-b59c16de268f") # provide artifact_uid
# delete_all()

# train_model()
# client.training.list()
# definition_details = client.repository.get_definition_details()
# client.repository.list_models()


# store_model("training--cukbB5mg") # provide run_guid
# deploy_model("00c8653e-3022-4942-934f-c53dd287eec1") # provide model_uid
# get_model_details("9e39aec2-3385-4b20-bda5-fe41f7dc4e9f")

# retrain_model("1162d2a2-f0fb-4293-9e25-2327e7017e79") # Provide definition_uid
# client.repository.list_models()
# update_model("00c8653e-3022-4942-934f-c53dd287eec1")
# client.deployments.list()
# update_deployment("ed7c91b1-79b5-4e2a-bc4e-1fc5bf16f1f0")

# print(client.repository.get_model_details())
# client.repository.FunctionMetaNames.get()
