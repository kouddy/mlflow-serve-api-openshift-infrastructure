import mlflow
import mlflow.sklearn
import numpy as np
from flask import Flask, request

app = Flask(__name__)

@app.route('/stress')
def stress():
    mlflow.set_tracking_uri("http://mlflow-tracking-server-route-anomaly-detection-in-it-systems.apps.us-east-1.starter.openshift-online.com")
    model = mlflow.sklearn.load_model("models:/QualityOfLife/Production")

    bmi = float(request.args.get('bmi'))
    vac = float(request.args.get('vac'))
    shout = float(request.args.get('shout'))

    
    stress_level = model.predict(np.array([bmi, vac, shout]).reshape(1, -1))[0]
    if stress_level>2.5:
        return "You are at risk of experiencing HIGH levels of stress daily"
    elif (stress_level<2.5) & (stress_level>1):
        return "You are at risk of experiencing MODERATE levels of stress daily"
    elif (stress_level<1):
        return "You are at risk of experiencing SOME levels of stress daily"
    