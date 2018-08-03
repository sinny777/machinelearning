
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
    # client.training.list()
    # run_guid = 'training-JSFYSi5mg'
    meta_props = {"name": "MyCarNLC_ML_Model", "frameworkName": "tensorflow"}
    # model_path = "results/my_nlc_model.h5"
    saved_model_details = client.repository.store_model(run_guid, meta_props)
    print(saved_model_details)

def deploy_model(model_uid):
    # model_uid = "7fa84241-a080-4000-b2e9-c4a009861476"
    deployment_details = client.deployments.create(model_uid, "MyCarNLC_ML_Model_Deploy")
    scoring_url = client.deployments.get_scoring_url(deployment_details)
    print(scoring_url)

def delete_model(artifact_uid):
    # artifact_uid = "9cd7108e-1310-4a72-8011-b59c16de268f"
    client.repository.delete_model(artifact_uid)

# client.repository.list_models()
# delete_model()
# train_model()
# store_model()
# deploy_model()
client.deployments.list()
# client.training.list()
# print(client.repository.get_model_details())
