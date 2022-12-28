import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import pandas as pd
import numpy as np
import json

app = Flask(__name__)
## load the model 
lr_model = pickle.load(open('lr1.pkl', 'rb'))
scalar = pickle.load(open('sc.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    if request.method == 'POST':
        data=request.json['data']
        to_predict_list= list(data.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)
        if int(result) == 1:
            prediction = 'Esta persona puede acceder a un préstamo'
        else:
            prediction = 'Esta persona no puede acceder a un préstamo'
        return prediction

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,-1)
    to_predict = scalar.transform(to_predict)
    output=lr_model.predict(to_predict)
    return output[0]
    

if __name__=="__main__":
    app.run(debug=True)
