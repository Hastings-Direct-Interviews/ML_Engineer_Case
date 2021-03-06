import os, json
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core.model import Model
import azureml.core
from azureml.core import Run
from azureml.core.authentication import AzureCliAuthentication
cli_auth = AzureCliAuthentication()

# Get workspace
ws = Workspace.from_config(auth=cli_auth)

# Get the latest run_id
with open("aml_config/run_id.json") as f:
    config = json.load(f)

print("Print config")
print(config)

new_model_run_id = config["run_id"]
experiment_name = config["experiment_name"]
exp = Experiment(workspace=ws, name=experiment_name)

print("New model run id:{}".format(new_model_run_id))


try:
    # Get most recently registered model, we assume that is the model in production. Download this model and compare it with the recently trained model by running test with same data set.
    model_list = Model.list(ws)
    production_model = next(
        filter(
            lambda x: x.created_time == max(model.created_time for model in model_list),
            model_list,
        )
    )

    print("Model List")
    print(model_list)

    production_model_run_id = production_model.tags.get("run_id")
    run_list = exp.get_runs()

    # Get the run history for both production model and newly trained model and compare mse
    production_model_run = Run(exp, run_id=production_model_run_id)
    new_model_run = Run(exp, run_id=new_model_run_id)

    production_model_mae = production_model_run.get_metrics().get("mae")
    new_model_mae = new_model_run.get_metrics().get("mae")
    print(
        "Current Production model mae: {}, New trained model mae: {}".format(
            production_model_mae, new_model_mae
        )
    )

    promote_new_model = False
    if new_model_mae < production_model_mae:
        promote_new_model = True
        print("New trained model performs better, thus it will be registered")
except:
    promote_new_model = True
    print("This is the first model to be trained, thus nothing to evaluate for now")

run_id = {}
run_id["run_id"] = ""
# Writing the run id to /aml_config/run_id.json
if promote_new_model:
    run_id["run_id"] = new_model_run_id

run_id["experiment_name"] = experiment_name
with open("aml_config/run_id.json", "w") as outfile:
    json.dump(run_id, outfile)


print(run_id)