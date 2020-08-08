import joblib
from flask import jsonify


class modelSelection:
    def __init__(self, model_name):
        self.model_name = model_name

    def select_model(self):
        path = "trainedmodels/"
        if self.model_name == "xgb":
            jl_file = "xgboost_jl.pkl"
            model = joblib.load(open(path+jl_file, 'rb'))
        elif self.model_name == "lda":
            jl_file = "lda_jl.pkl"
            model = joblib.load(open(path+jl_file, 'rb'))
        elif self.model_name == "rf":
            jl_file = "balancedrf_jl.pkl"
            model = joblib.load(open(path+jl_file, 'rb'))
        elif self.model_name == "dt":
            jl_file = "decisiontree_jl.pkl"
            model = joblib.load(open(path+jl_file, 'rb'))
        elif self.model_name == "gbm":
            jl_file = "gbm_jl.pkl"
            model = joblib.load(open(path+jl_file, 'rb'))
        else:
            return "Invalid Model Selection"

        return model
