#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 15:57:34 2019

@author: rgupt100
"""

# Importing libraries
from flask import Flask, jsonify, request
from sklearn.externals import joblib
import numpy as np

app = Flask(__name__)

#Load the model

stored_model = joblib.load('random_forest.mdl')
sex_map = joblib.load('sex_map.file')
title_map = joblib.load('title_map.file')
    
@app.route('/')
def main():
    return "This is the main page"

@app.route('/predict', methods=['GET'])
def predict():
# Predict the probability of survival
    args = request.args
    #required_args = ['class', 'sex', 'age', 'sibsp', 'parch', 'title']
    input_values = np.array([args['class'], sex_map[args['sex']], 
                            args['age'], args['sibsp'], 
                            args['parch'], title_map[args['title'].lower()]]).reshape(1, -1)
    probability = stored_model.predict_proba(input_values)[:, 1][0]
    return jsonify({'probabilityOfSurvival': probability})

if __name__ == '__main__':
    app.run(host='0.0.0.0')