"""
Copyright (C) Microsoft Corporation. All rights reserved.​
 ​
Microsoft Corporation (“Microsoft”) grants you a nonexclusive, perpetual,
royalty-free right to use, copy, and modify the software code provided by us
("Software Code"). You may not sublicense the Software Code or any use of it
(except to your affiliates and to vendors to perform work on your behalf)
through distribution, network access, service agreement, lease, rental, or
otherwise. This license does not purport to express any claim of ownership over
data you may have shared with Microsoft in the creation of the Software Code.
Unless applicable law gives you more rights, Microsoft reserves all other
rights not expressly granted herein, whether by implication, estoppel or
otherwise. ​
 ​
THE SOFTWARE CODE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
MICROSOFT OR ITS LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THE SOFTWARE CODE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
import os
import json
import sys
import azureml.core
from azureml.core import Workspace, Experiment, Run
from azureml.core.model import Model
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.webservice import AciWebservice, Webservice
import mlflow
import mlflow.azureml

# Load the JSON settings file and relevant section
print("Loading settings")
with open(os.path.join("code", "settings.json")) as f:
    settings = json.load(f)
deployment_settings = settings["deployment"]

# Get details from Run
print("Loading Run Details")
with open(os.path.join("code", "run_details.json")) as f:
    run_details = json.load(f)

# Get workspace
print("Loading Workspace")
cli_auth = AzureCliAuthentication()
config_file_path = os.environ.get("GITHUB_WORKSPACE", default="aml_service")
config_file_name = "aml_arm_config.json"
ws = Workspace.from_config(
    path=config_file_path,
    auth=cli_auth,
    _file_name=config_file_name)
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep='\n')


# -----------mlflow----------------------------------
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


# ---------------------Build image ----------------------------------------
# Use MLflow to build a Container Image for the trained model
# Use the `mlflow.azuereml.build_image` function to build an Azure Container Image for the trained MLflow model.
# This function also registers the MLflow model with a specified Azure ML workspace.
# The resulting image can be deployed to Azure Container Instances (ACI) or Azure Kubernetes Service (AKS) for real-time serving.

model_image, azure_model = mlflow.azureml.build_image(model_uri='runs:/{}/{}'.format(run_details["run_id"], deployment_settings["model"]["path"]),
                                                      workspace=ws,
                                                      model_name=deployment_settings["model"]["name"],
                                                      image_name=deployment_settings["model"]["name"],
                                                      description="Sklearn image for predicting iris type",
                                                      synchronous=False)
model_image.wait_for_creation(show_output=True)


# ------------------------ Create an ACI webservice deployment---------------------
# The [ACI platform](https://docs.microsoft.com/en-us/azure/container-instances/) is the recommended environment for staging and developmental model deployments.
# Using the Azure ML SDK, deploy the Container Image for the trained MLflow model to ACI.


dev_webservice_name = "iris-model"
dev_webservice_deployment_config = AciWebservice.deploy_configuration()
dev_webservice = Webservice.deploy_from_image(
    name=dev_webservice_name, image=model_image, deployment_config=dev_webservice_deployment_config, workspace=ws)
dev_webservice.wait_for_deployment()
