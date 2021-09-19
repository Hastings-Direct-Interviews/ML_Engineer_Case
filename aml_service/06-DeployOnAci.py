import json, datetime, sys
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.webservice import AciWebservice
from azureml.core.authentication import AzureCliAuthentication
from azureml.core import Environment
from azureml.core.environment import CondaDependencies
from azureml.core.model import InferenceConfig


cli_auth = AzureCliAuthentication()
# Get workspace
ws = Workspace.from_config(auth=cli_auth)# Get the Image to deploy details

try:
    with open("aml_config/model.json") as f:
        config = json.load(f)
except:
    print("No new model, thus no deployment on ACI")
    # raise Exception('No new model to register as production model perform better')
    sys.exit(0)

att_model = ws.models["fnol_attritional_model.cbm"]
large_model = ws.models["fnol_large_claim_propensity_model.cbm"]
large_severity = ws.models["large_severity.json"]
model_meta_data = ws.models["model_meta_data.json"]


# Create an environment
print("Creating Environment")
conda_dependencies_file_path = "aml_config/conda_dependencies.yml"

# Combining scoring script and environment
print("Combining scoring script and environment")
inference_config = InferenceConfig(entry_script='code/scoring/score.py',
                                    conda_file=conda_dependencies_file_path
                                    
                                    )


# Define deployment configuration
print("Define deployment configuration")

aci_config = AciWebservice.deploy_configuration(
    cpu_cores=1,
    memory_gb=1,
#    tags={"area": "fnol", "type": "regression"},
#    description="fnol model",
)

# Deploy the model
print("Deploy the model")

aci_service_name = "aciwebservice" + datetime.datetime.now().strftime("%m%d%H")

service = Model.deploy(workspace=ws,
                        name=aci_service_name,
                        models=[att_model, large_model, large_severity, model_meta_data],
#                        models=[],                        
                        inference_config=inference_config,
                        deployment_config=aci_config                       
                        )
print("Wait for deployment")
service.wait_for_deployment(show_output=True)

print(
    "Deployed ACI Webservice: {} \nWebservice Uri: {}".format(
        service.name, service.scoring_uri
    )
)

# Writing the ACI details to /aml_config/aci_webservice.json
aci_webservice = {}
aci_webservice["aci_name"] = service.name
aci_webservice["aci_url"] = service.scoring_uri
with open("aml_config/aci_webservice.json", "w") as outfile:
    json.dump(aci_webservice, outfile)