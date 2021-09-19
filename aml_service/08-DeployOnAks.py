import json, datetime, sys
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.webservice import AksWebservice
from azureml.core.authentication import AzureCliAuthentication
from azureml.core import Environment
from azureml.core.environment import CondaDependencies
from azureml.core.model import InferenceConfig
from azureml.core.compute import ComputeTarget, AksCompute


cli_auth = AzureCliAuthentication()
# Get workspace
ws = Workspace.from_config(auth=cli_auth)# Get the Image to deploy details

try:
    with open("aml_config/model.json") as f:
        config = json.load(f)
except:
    print("No new model, thus no deployment on AKS")
    # raise Exception('No new model to register as production model perform better')
    sys.exit(0)

att_model = ws.models["fnol_attritional_model.cbm"]
large_model = ws.models["fnol_large_claim_propensity_model.cbm"]
large_severity = ws.models["large_severity.json"]
model_meta_data = ws.models["model_meta_data.json"]

# Create an environment
print("Creating Environment")
conda_dependencies_file_path = "aml_config/conda_dependencies.yml"

# fnol_env_name = "fnol-env"
# fnolenv = Environment.get(workspace=ws, name='AzureML-Minimail').clone(fnol_env_name)
# conda_dep = CondaDependecies(conda_dependencies_file_path)
# fnolenv.python.conda_dependencies=conda_dep

# Combining scoring script and environment
print("Combining scoring script and environment")
inference_config = InferenceConfig(runtime= "python",
                                    entry_script='code/scoring/score.py',
                                    conda_file=conda_dependencies_file_path
                                    )

# Create a production cluster for the AKS service
print("Creating a production cluster for the AKS service")

# Check if AKS already Available
try:
    with open("aml_config/aks_webservice.json") as f:
        config = json.load(f)
    cluster_name = config["aks_cluster_name"]
    aks_service_name = config["aks_service_name"]
    compute_list = ws.compute_targets()
    production_cluster, = (c for c in compute_list if c.name == cluster_name)

except:
    cluster_name = "aks" + datetime.datetime.now().strftime("%m%d%H")
    aks_service_name = "akswebservice" + datetime.datetime.now().strftime("%m%d%H")
    compute_config = AksCompute.provisioning_configuration(location="uksouth", agent_count=6, vm_size="Standard_F4")
    print(
        "No AKS found in aks_webservice.json. Creating new Aks: {} and AKS Webservice: {}".format(
            cluster_name, aks_service_name
        )
    )
    # Create the cluster
    production_cluster = ComputeTarget.create(
        workspace=ws, name=cluster_name, provisioning_configuration=compute_config
    )

    production_cluster.wait_for_completion(show_output=True)
    print(production_cluster.provisioning_state)
    print(production_cluster.provisioning_errors)

# Define deployment configuration
print("Define deployment configuration")

aks_config = AksWebservice.deploy_configuration(
    cpu_cores=1,
    memory_gb=1,
    # tags={"area": "fnol", "type": "regression"},
    # description="fnol model",
)

# Deploy the model
print("Deploy the model")

service = Model.deploy(workspace=ws,
                        name=aks_service_name,
                        models=[att_model, large_model, large_severity, model_meta_data],
                        inference_config=inference_config,
                        deployment_config=aks_config,
                        deployment_target=production_cluster                    
                        )

service.wait_for_deployment(show_output=True)
print(service.state)

print(
    "Deployed AKS Webservice: {} \nWebservice Uri: {}".format(
        service.name, service.scoring_uri
    )
)

# Writing the AKS details to /aml_config/aks_webservice.json
aks_webservice = {}
aks_webservice["aks_cluster_name"] = cluster_name
aks_webservice["aks_service_name"] = service.name
aks_webservice["aks_url"] = service.scoring_uri
aks_webservice["aks_keys"] = service.get_keys()
with open("aml_config/aks_webservice.json", "w") as outfile:
    json.dump(aks_webservice, outfile)