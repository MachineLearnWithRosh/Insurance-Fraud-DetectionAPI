from operator import itemgetter

import numpy as np
from flask import Flask, request, jsonify, render_template, Response
from data_transformation.dataTransformation import dataTransformation
from trainedmodels.selectModel import modelSelection
from flask_cors import CORS, cross_origin
import os, joblib
from sklearn.preprocessing import StandardScaler

"""
Developed By : Roshan Kumar Gupta
Date: 31st July 2020
"""

app = Flask(__name__)
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')


@app.route('/')
def home():
    return jsonify(Welcome='Welcome to Claim Price Prediction App')


def standardized_data(data):
    path = "trainedmodels/"
    ss_file = "standardscalar.pkl"
    scalar = joblib.load(open(path + ss_file, 'rb'))
    data = scalar.transform(data)
    return data


@app.route('/predict_api/<modelname>', methods=['POST'])
def predict_api(modelname):
    """
    For direct api calls
    """
    try:
        if request.method == 'POST':
            response = request.get_json(force=True)
            trans = dataTransformation(response)

            # selecting models
            select_model = modelSelection(modelname)
            model = select_model.select_model()
            if model == "Invalid Model Selection":
                return jsonify(status="Invalid Model Selection, Please Verify the ModelName"), 404

            # encoding categorical features
            data = trans.map_categorical_data()
            if data == "Invalid Input":
                return jsonify(status="Invalid input, Please check the fields"), 422

            # standardizing the data
            data_curated = standardized_data([np.array(list(data.values()))])

            # classifying the output
            prediction = model.predict(data_curated)

            feat_imp = trans.get_feat_importance(model)
            feat_imp = dict(feat_imp[:10].values.tolist())
            pred_amt = round(float(prediction), 2)
            print(type(data_curated))
            print(type(np.array([pred_amt])))

            model_score = model.score(data_curated, np.array([pred_amt]))

            print(model_score)

            # print(dict(sorted(feat_imp.items(), key=itemgetter(1), reverse=True)[:10]))

            return jsonify(claim_amount=pred_amt,
                           intepretation=dict(sorted(feat_imp.items(), key=itemgetter(1), reverse=True)[:10]))

    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)


if __name__ == "__main__":
    app.run(debug=True)
