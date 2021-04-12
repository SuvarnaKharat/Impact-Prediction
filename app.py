# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:21:08 2020

@author: User
"""

from flask import Flask,jsonify,render_template,request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
clf=pickle.load(open('Incident_management_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/impact',methods=['POST'])
def impact():
    if request.method == 'POST': 
     features=[val for val in request.form.values()]
     prediction = clf.predict(features)
     
     return render_template('index.html', prediction_text='Impact will be {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)