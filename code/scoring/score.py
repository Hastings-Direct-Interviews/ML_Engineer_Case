import json
import numpy as np
from azureml.core.model import Model
from catboost import CatBoostRegressor, CatBoostClassifier, Pool


def init():
    print("Start Initiation")
    global att_model
    global large_model
    global large_severity
    global feature_names
    global cat_features

    print("Load Models")

    att_model_path = Model.get_model_path(model_name="fnol_attritional_model.cbm")
    large_model_path = Model.get_model_path(model_name="fnol_large_claim_propensity_model.cbm")
    large_severity_path = Model.get_model_path(model_name="large_severity.json")
    model_meta_data_path = Model.get_model_path(model_name="model_meta_data.json")
    
    # load the model from file into a global object
    att_model = CatBoostRegressor()
    att_model.load_model(att_model_path)

    large_model = CatBoostClassifier()
    large_model.load_model(large_model_path)

    with open(large_severity_path) as f:
        large_severity_json = json.load(f)

    large_severity = large_severity_json["large_severity"]

    with open(model_meta_data_path) as f:
        model_meta_data_json = json.load(f)

    feature_names = model_meta_data_json["feature_names"]
    cat_features = model_meta_data_json["cat_features"]

def run(raw_data):
    try:
        data = json.loads(raw_data)["data"]
        data = np.array(data)

        pred_pool = Pool(
            data = data,
            feature_names = feature_names,
            cat_features = cat_features
        )

        att_model_preds = att_model.predict(pred_pool)
        large_model_preds = large_model.predict_proba(pred_pool)[:, 1]
        result = att_model_preds + (large_model_preds * large_severity)

        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})


# if __name__ == "__main__":
#     # Test scoring
#     init()
#     test_row = '{"data":[[1,2,3,4,5,6,7,8,9,10],[10,9,8,7,6,5,4,3,2,1]]}'
#     prediction = run(test_row)
#     print("Test result: ", prediction)