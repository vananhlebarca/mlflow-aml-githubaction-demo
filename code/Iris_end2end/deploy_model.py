

from azureml.core.webservice import AciWebservice, Webservice
import os
import sys
import json
from azureml.core import Workspace, Image, Experiment, Run
from azureml.core.webservice import Webservice, AciWebservice
from azureml.exceptions import WebserviceException
from azureml.core.authentication import AzureCliAuthentication
import mlflow
import mlflow.azureml


# Load the JSON settings file and relevant sections
print("Loading settings")
with open(os.path.join("code", "settings.json")) as f:
    settings = json.load(f)
deployment_settings = settings["deployment"]
aci_settings = deployment_settings["dev_deployment"]

# Get details from Run
print("Loading Run Details")
with open(os.path.join("code", "run_details.json")) as f:
    run_details = json.load(f)


# Get workspace
print("Loading Workspace")
cli_auth = AzureCliAuthentication()
config_file_path = os.environ.get("GITHUB_WORKSPACE", default="code")
config_file_name = "aml_arm_config.json"
ws = Workspace.from_config(
    path=config_file_path,
    auth=cli_auth,
    _file_name=config_file_name)
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep='\n')

# -----------mlflow----------------
# Load the JSON settings file and relevant section
print("Loading settings")
with open(os.path.join("code", "settings.json")) as f:
    settings = json.load(f)
experiment_settings = settings["experiment"]
deployment_settings = settings["deployment"]

os.environ["MLFLOW_TRACKING_URI"] = ws.get_mlflow_tracking_uri()

# Attach Experiment
print("Loading Experiment")
exp = Experiment(workspace=ws, name=experiment_settings["name"])
mlflow.set_experiment(exp.name)
print(exp.name, exp.workspace.name, sep="\n")

# --------------------------------------------------


# Deploying model on ACI
print("Deploying model on ACI")
aci_config = AciWebservice.deploy_configuration(cpu_cores=2,
                                                memory_gb=5)
# Deploying dev web service from run
dev_service = mlflow.azureml.deploy(model_uri='runs:/{}/{}'.format(run_details["run_id"], deployment_settings["model"]["path"]),
                                    workspace=ws,
                                    deployment_config=aci_config,
                                    service_name="iris-aci")
#                                          model_name=deployment_settings["model"]["name"])
