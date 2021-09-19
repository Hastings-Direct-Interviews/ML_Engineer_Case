from azureml.core import Workspace
from azureml.core import Dataset


#from azureml.core.runconfig import RunConfiguration
from azureml.core.authentication import AzureCliAuthentication
cli_auth = AzureCliAuthentication()

# Get workspace
ws = Workspace.from_config(auth=cli_auth)

default_ds = ws.get_default_datastore()

print(default_ds)

default_ds.upload_files(files=['./data/Data_Scientist_Interview_Task.csv'], # Upload the diabetes csv files in /data
                       target_path='fnol-data/', # Put it in a folder path in the datastore
                       overwrite=True, # Replace existing files of the same name
                       show_progress=True)

tab_data_set = Dataset.Tabular.from_delimited_files(path=(default_ds, 'fnol-data/Data_Scientist_Interview_Task.csv'))

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


