import urllib3, requests, json

# retrieve your wml_service_credentials_username, wml_service_credentials_password, and wml_service_credentials_url from the
# Service credentials associated with your IBM Cloud Watson Machine Learning Service instance

with open('config.json', 'r') as f:
    global SECRET_CONFIG
    SECRET_CONFIG = json.load(f)

wml_credentials = SECRET_CONFIG["wml_credentials"]

headers = urllib3.util.make_headers(basic_auth='{username}:{password}'.format(username=wml_credentials['username'], password=wml_credentials['password']))
url = '{}/v3/identity/token'.format(wml_credentials['url'])
response = requests.get(url, headers=headers)
mltoken = json.loads(response.text).get('token')

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

# NOTE: manually define and pass the array(s) of values to be scored in the next line
payload_scoring = {"values": ["HOw old are you", "turn on the kitchen fan"]}

response_scoring = requests.post('https://ibm-watson-ml.mybluemix.net/v3/wml_instances/e7e44faf-ff8d-4183-9f37-434e2dcd6852/deployments/82674972-5185-4f8b-8096-fd7c83264567/online', json=payload_scoring, headers=header)
print("Scoring response: >> ")
print(json.loads(response_scoring.text))
