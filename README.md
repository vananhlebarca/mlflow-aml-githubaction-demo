# An end2end MLOPs workflow using MLflow and AzureML:
- MLflow tracks metrics, parameters and logs the model to Azure ML experiment.
- The model is then registered and deploy as an ACI service in Azure ML

# Github Action automates the workflow:
- create AML workspace

There are total 4 options for running the workflow: Make change in .github/workflow/train-deploy.yml to choose one of them
