#import os, json, sys
from azureml.core import Workspace
from azureml.core import Dataset


#from azureml.core import Run
#rom azureml.core import Experiment
#from azureml.core.model import Model

#from azureml.core.runconfig import RunConfiguration
from azureml.core.authentication import AzureCliAuthentication
cli_auth = AzureCliAuthentication()

# Get workspace
ws = Workspace.from_config(auth=cli_auth)

default_ds = ws.get_default_datastore()

print(default_ds)

tab_data_set = Dataset.Tabular.from_delimited_files(path=(default_ds, './data/Data_Scientist_Interview_Task.csv'))

# Register the tabular dataset
try:
    tab_data_set = tab_data_set.register(workspace=ws, 
                                        name='fnol dataset',
                                        description='fnol data',
                                        tags = {'format':'CSV'},
                                        create_new_version=True)
except Exception as ex:
    print(ex)

print('Dataset registered')


