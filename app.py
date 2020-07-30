import numpy as np
from flask import Flask, request, jsonify, render_template, Response
import joblib
from data_transformation.dataTransformation import dataTransformation
from flask_cors import CORS, cross_origin
import os
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')


@app.route('/')
def home():
    return jsonify(Welcome='Welcome to Fraud detection App')


def select_model(model_name):
    path ="trainedmodels/"
    if model_name == "xgb":
        jl_file = "xgboost_jl.pkl"
        model = joblib.load(open(path+jl_file, 'rb'))
    elif model_name == "lda":
        jl_file = "lda_jl.pkl"
        model = joblib.load(open(path+jl_file, 'rb'))
    elif model_name == "rf":
        jl_file = "balancedrf_jl.pkl"
        model = joblib.load(open(path+jl_file, 'rb'))
    else:
        return jsonify({'response': 'Invalid Model selection'})
    return model


def standardized_data(data):
    path = "trainedmodels/"
    ss_file = "standardscalar.pkl"
    scalar = joblib.load(open(path+ss_file, 'rb'))
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
            data = trans.map_categorical_data()
            model = select_model(modelname)
            data_curated = standardized_data([np.array(list(data.values()))])
            prediction = model.predict(data_curated)

            if prediction[0] == 0:
                status = "Fraud Not Detected"
            elif prediction[0] == 1:
                status = "Fraud Detected"

            score = model.predict_proba(data_curated)[0][1]
            print(type(score))

            return jsonify(Status=status, Score=str(score) + "%")
            #return jsonify({'status': 'Working perfectly fine'})

    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)


if __name__ == "__main__":
    app.run(debug=True)
