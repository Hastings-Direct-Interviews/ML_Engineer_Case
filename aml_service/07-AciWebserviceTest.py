import json, sys
from azureml.core import Workspace
from azureml.core.webservice import Webservice
from azureml.core.authentication import AzureCliAuthentication

cli_auth = AzureCliAuthentication()
# Get workspace
ws = Workspace.from_config(auth=cli_auth)
# Get the ACI Details
try:
    with open("aml_config/aci_webservice.json") as f:
        config = json.load(f)
except:
    print("No new model, thus no deployment on ACI")
    sys.exit(0)

service_name = config["aci_name"]
# Get the hosted web service
service = Webservice(name=service_name, workspace=ws)

# Input for Model with all features
input_j = [list(range(1, 38)), list(reversed(range(1, 38)))]
print(input_j)
test_sample = json.dumps({"data": input_j})
test_sample = bytes(test_sample, encoding="utf8")
try:
    prediction = service.run(input_data=test_sample)
    print(prediction)
except Exception as e:
    result = str(e)
    print(result)
    raise Exception("ACI service is not working as expected")
