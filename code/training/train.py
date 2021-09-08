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
from sklearn.metrics import mean_absolute_error, log_loss


run = Run.get_submitted_run()

print("Running train.py")

print("Load Data")
df = run.input_datasets['training_data'].to_pandas_dataframe()

###### Data Cleaning ######

print("Cleaning data")

# drop columns that don't provide useful information
df = df.drop(columns=['Claim Number', 'Notifier', 'Loss_code', 'Loss_description', 'Inception_to_loss'])

# drop additional columns that we won't use in this round of modelling
df = df.drop(columns=['date_of_loss', 'Time_hour'])

# clean-up missing values
df['Weather_conditions'] = df['Weather_conditions'].fillna('N/K')
df['PH_considered_TP_at_fault'] = df['PH_considered_TP_at_fault'].replace('#', 'n/k')

# set values in target variables less than zero to zero
df.loc[df['Incurred'] < 0, ['Incurred', 'Capped Incurred']] = 0

###### Build Attritional Model ######

print("Training attritional model")

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

params = {"objective": "Tweedie:variance_power=1.99",
          "iterations": 1000,
          "random_seed": 69,
          "verbose": False}

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

# upload the model file explicitly into artifacts
run.upload_file(name="./outputs/" + model_name, path_or_stream=model_name)
print("Uploaded the model {} to experiment {}".format(model_name, run.experiment.name))
dirpath = os.getcwd()
print(dirpath)


###### Build Large Claim Propensity Model ######

print("Training large claim propensity model")

# Create large claim target variable
df['Large_Prop'] = np.where(df['Incurred'] > df['Capped Incurred'], 1, 0)

# create a data frame for the large claims model
df_large = df.drop(columns=['Incurred', 'Capped Incurred'])

# Cross validation to find optimal number of features
feature_names = list(df_large.drop(columns=['Large_Prop']))
cat_features = df_large.drop(columns=['Large_Prop']).select_dtypes(include=['object']).columns.tolist()

data_pool = Pool(
    data = df_large.drop(columns=['Large_Prop']),
    label = df_large['Large_Prop'],
    feature_names = feature_names,
    cat_features = cat_features
)

params = {"objective": "Logloss",
          "iterations": 1000,
          "random_seed": 69,
          "verbose": False}

scores = cv(pool = data_pool,
            params = params,
            fold_count = 4,
            early_stopping_rounds = 15)

optimal_iterations = len(scores)-15

# Fit final model
params = {"objective": "Logloss",
          "iterations": optimal_iterations,
          "random_seed": 69,
          "verbose": False}


large_model = CatBoostClassifier(**params)

large_model.fit(data_pool)

# calculate logloss and log
large_model_preds = large_model.predict_proba(data_pool)[:, 1]

logloss = log_loss(df_large['Large_Prop'], large_model_preds)

run.log("logloss", log_loss(df_large['Large_Prop'], large_model_preds))

# save the model
model_name = "fnol_large_claim_propensity_model.cbm"

large_model.save_model(model_name, format="cbm")

# upload the model file explicitly into artifacts
run.upload_file(name="./outputs/" + model_name, path_or_stream=model_name)
print("Uploaded the model {} to experiment {}".format(model_name, run.experiment.name))
dirpath = os.getcwd()
print(dirpath)


###### Build Large Claim Severity Model ######

print("Calculate Large Loss Severity")

# Create large claim severity target variable
df['Large_Incurred'] = df['Incurred'] - df['Capped Incurred']

# Calculate large claim severity
large_severity = df[df['Large_Prop']==1]['Large_Incurred'].mean()

print("Following files are uploaded ")
print(run.get_file_names())
run.complete()