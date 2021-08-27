# import pickle
# from azureml.core import Workspace

# import os
# from sklearn.datasets import load_diabetes
# from sklearn.linear_model import Ridge
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split
# from sklearn.externals import joblib
# import numpy as np
# import json
# import subprocess
# from typing import Tuple, List


from azureml.core.run import Run
import pandas as pd
from catboost import CatBoostRegressor, Pool, cv
from sklearn.metrics import mean_absolute_error


run = Run.get_submitted_run()

print("Running train.py")

df = pd.read_csv('./data/Data_Scientist_Interview_Task.csv')

###### Data Cleaning ######

# drop columns that don't provide useful information
df = df.drop(columns=['Claim Number', 'Notifier', 'Loss_code', 'Loss_description', 'Inception_to_loss'])

# drop additional columns that we won't use in this round of modelling
df = df.drop(columns=['date_of_loss', 'Time_hour'])

# clean-up missing values
df['Weather_conditions'] = df['Weather_conditions'].fillna('N/K')
df['PH_considered_TP_at_fault'] = df['PH_considered_TP_at_fault'].replace('#', 'n/k')

###### Build Attritional Model ######

# create a data frame for the attritional claims model
df_att = df.drop(columns=['Incurred'])

# Cross validation to find optimal number of features
feature_names = list(df_att.drop(columns=['Capped Incurred']))
cat_features = df_att.drop(columns=['Capped Incurred']).select_dtypes(include=['object']).columns.tolist()

data_pool = Pool(
    data = df_att.drop(columns=['Capped Incurred']),
    label = df_att['Capped Incurred'],
    feature_names = feature_names,
    cat_features = cat_features
)

scores = cv(pool = data_pool,
            params = params,
            fold_count = 4,
            early_stopping_rounds = 15)

optimal_iterations = len(scores)-15

# Fit final model
params = {"objective": "Tweedie:variance_power=1.99",
          "iterations": optimal_iterations,
          "random_seed": 69,
          "verbose": False}


att_model = CatBoostRegressor(**params)

att_model.fit(data_pool)

# calculate mae and log

att_model_preds = att_model.predict(data_pool)

run.log("mae", mean_absolute_error(att_model_preds, df_att['Capped Incurred']))

# save the model
model_name = "fnol_attritional_model.cbm"

att_model.save_model(model_name, format="cbm")





# Save model as part of the run history
model_name = "sklearn_regression_model.pkl"
# model_name = "."

with open(model_name, "wb") as file:
    joblib.dump(value=reg, filename=model_name)

# upload the model file explicitly into artifacts
run.upload_file(name="./outputs/" + model_name, path_or_stream=model_name)
print("Uploaded the model {} to experiment {}".format(model_name, run.experiment.name))
dirpath = os.getcwd()
print(dirpath)



print("Following files are uploaded ")
print(run.get_file_names())
run.complete()