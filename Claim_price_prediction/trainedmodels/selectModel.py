import joblib
from flask import jsonify


class modelSelection:
    def __init__(self, model_name):
        self.model_name = model_name

    def select_model(self):
        path = "trainedmodels/"
        if self.model_name == "xgb":
            jl_file = "xgb.pkl"
            model = joblib.load(open(path+jl_file, 'rb'))
        elif self.model_name == "rf":
            jl_file = "randomforest.pkl"
            model = joblib.load(open(path+jl_file, 'rb'))
        elif self.model_name == "dt":
            jl_file = "dt.pkl"
            model = joblib.load(open(path+jl_file, 'rb'))
        else:
            return "Invalid Model Selection"

        return model
