import os, json, sys
from azureml.core import Workspace
from azureml.core import Run
from azureml.core import Experiment
from azureml.core.model import Model

from azureml.core.runconfig import RunConfiguration
from azureml.core.authentication import AzureCliAuthentication
cli_auth = AzureCliAuthentication()

# Get workspace
ws = Workspace.from_config(auth=cli_auth)

# Get the latest evaluation result
try:
    with open("aml_config/run_id.json") as f:
        config = json.load(f)
    if not config["run_id"]:
        raise Exception("No new model to register as production model performed better")
except:
    print("No new model to register as production model performed better")
    sys.exit(0)

run_id = config["run_id"]
experiment_name = config["experiment_name"]
exp = Experiment(workspace=ws, name=experiment_name)

run = Run(experiment=exp, run_id=run_id)
names = run.get_file_names
names()
print("Run ID for last run: {}".format(run_id))
model_local_dir = "model"
os.makedirs(model_local_dir, exist_ok=True)

# Download Models to Project root directory
model_name_1 = "fnol_attritional_model.cbm"
run.download_file(
    name="./outputs/" + model_name_1, output_file_path="./model/" + model_name_1
)
print("Downloaded model {} to Project root directory".format(model_name_1))
os.chdir("./model")
model_1 = Model.register(
    model_path=model_name_1,  # this points to a local file
    model_name=model_name_1,  # this is the name the model is registered as
    tags={"area": "FNOL", "type": "regression", "run_id": run_id},
    description="Attritional claims model for FNOL",
    workspace=ws,
)
os.chdir("..")
print(
    "Model registered: {} \nModel Description: {} \nModel Version: {}".format(
        model_1.name, model_1.description, model_1.version
    )
)
model_name_2 = "fnol_large_claim_propensity_model.cbm"
run.download_file(
    name="./outputs/" + model_name_2, output_file_path="./model/" + model_name_2
)
print("Downloaded model {} to Project root directory".format(model_name_2))
os.chdir("./model")
model_2 = Model.register(
    model_path=model_name_2,  # this points to a local file
    model_name=model_name_2,  # this is the name the model is registered as
    tags={"area": "FNOL", "type": "regression", "run_id": run_id},
    description="Large claims propensity model for FNOL",
    workspace=ws,
)
os.chdir("..")
print(
    "Model registered: {} \nModel Description: {} \nModel Version: {}".format(
        model_2.name, model_2.description, model_2.version
    )
)
model_name_3 = "large_severity.json"
run.download_file(
    name="./outputs/" + model_name_3, output_file_path="./model/" + model_name_3
)
print("Downloaded model {} to Project root directory".format(model_name_3))
os.chdir("./model")
model_3 = Model.register(
    model_path=model_name_3,  # this points to a local file
    model_name=model_name_3,  # this is the name the model is registered as
    tags={"area": "FNOL", "type": "regression", "run_id": run_id},
    description="Large claims severity model for FNOL",
    workspace=ws,
)
os.chdir("..")
print(
    "Model registered: {} \nModel Description: {} \nModel Version: {}".format(
        model_3.name, model_3.description, model_3.version
    )
)

# Remove the evaluate.json as we no longer need it
# os.remove("aml_config/evaluate.json")

# Writing the registered model details to /aml_config/model.json
model_json = {}
model_json["attritional_model_name"] = model_1.name
model_json["attritional_model_version"] = model_1.version
model_json["ll_prop_model_name"] = model_2.name
model_json["ll_prop_model_version"] = model_2.version
model_json["ll_sev_model_name"] = model_3.name
model_json["ll_sev_model_version"] = model_3.version
model_json["run_id"] = run_id
with open("aml_config/model.json", "w") as outfile:
    json.dump(model_json, outfile)

print(model_json)