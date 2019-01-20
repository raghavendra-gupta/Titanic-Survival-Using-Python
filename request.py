# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 15:57:34 2019

@author: rgupt100
"""

import requests

inputs = {
    'class': 2,
    'age': 22,
    'sibsp': 2,
    'parch': 0,
    'title': 'Mr',
    'sex': 'male',
}

url = 'http://localhost:5000/predict'
r = requests.get(url, inputs)
print(r.url)
print(r.json())