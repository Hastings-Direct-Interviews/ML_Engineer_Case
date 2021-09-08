#!/bin/bash

python --version
pip install azure-cli==2.2.0
pip install --upgrade azureml-sdk[cli]
pip install -r requirements.txt