# Custom Natural Language Classifier

This repository can be used as a reference to creating ML or Deep learning models using Tensorflow, Keras or any library of your choice.  This one has code for creating Natural Language classifier model that runs on IBM Watson Machine learning platform and can be configured to use runtime (CPU or GPU) as per your choice.  All the code to create and train the model is under the "build_code" folder.  Please note that whole of this code under "build_code" folder is zipped and deployed to IBM Watson Machine Learning platform and runs there.  

## Project Structure

  - build_code/IntentClassification.py: This is the main file that runs the code to create and train a model.  You can test it locally by running following command and passing a few parameters:

  ```sh
  python build_code/IntentClassification.py --data_dir data --result_dir results --config_file model_config.json --data_file data.csv
  ```
  - build_code/handlers: This has all the handlers like keras_model_handler.py, data_handler.py and cloud object storage handler (cos_handler.py)
  - utilities: This has one utility, but that's currently not been used.  You can remove it or can use it as per your requirement or preference.
  - Deployment.py: This file uses watson_machine_learning_client library to create definitions, train and store model, deploy model and use scoring endpoint.  This can be done by commenting or uncommenting a few lines of code for specific calls and then running following command:

  ```sh
  python Deployment.py --data_dir data --result_dir results --config_file model_config.json --data_file data.csv
  ```

  - config.json: This file is not part of the git commit and you need to create this file which should have following format with details of wml_credentials and cos_credentials:

  ```sh
  {
    "cos_credentials":{
                        "apikey": "-----------------------------------",
                        "cos_hmac_keys": {
                          "access_key_id": "-----------------------------------",
                          "secret_access_key": "-----------------------------------"
                        },
                        "endpoints": "https://cos-service.bluemix.net/endpoints",
                        "iam_apikey_description": "-----------------------------------",
                        "iam_apikey_name": "auto-generated-apikey------------------------------------",
                        "iam_role_crn": "crn:v1:bluemix:public:iam::::serviceRole:Writer",
                        "iam_serviceid_crn": "crn:v1:bluemix:public:iam-identity::a/-----------------------------------::serviceid:ServiceId------------------------------------",
                        "resource_instance_id": "crn:v1:bluemix:public:cloud-object-storage:global:a/-----------------------------------:-----------------------------------::"
                      },
    "wml_credentials":{
                        "apikey": "------------------------------------vpfRo3SBZrOajK",
                        "iam_apikey_description": "Auto generated apikey during resource-key operation for Instance - crn:v1:bluemix:public:pm-20:us-south:a/-----------------------------------::",
                        "iam_apikey_name": "auto-generated-apikey------------------------------------",
                        "iam_role_crn": "crn:v1:bluemix:public:iam::::serviceRole:Writer",
                        "iam_serviceid_crn": "crn:v1:bluemix:public:iam-identity::a/-----------------------------------::-----------------------------------",
                        "instance_id": "-----------------------------------",
                        "password": "-----------------------------------",
                        "url": "https://ibm-watson-ml.mybluemix.net",
                        "username": "-----------------------------------"
                      },
    "deployment_id": "-----------------------------------",
    "service_endpoint": "https://s3-api.us-geo.objectstorage.softlayer.net",
    "auth_endpoint": "https://iam.bluemix.net/oidc/token"
  }
```
