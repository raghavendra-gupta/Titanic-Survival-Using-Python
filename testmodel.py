#Import the Libraries
import pandas as pd
import re

#Importing and Reading the dataset
dataset = pd.read_csv(r"C:\Users\rgupt100\Documents\Personal Docs\Data_engineer_work_sample__1_-1\train.csv", header=0)

def cat_map(col_name):
    return dict([(cat, st_val) for st_val, cat in enumerate(col_name.cat.categories, start=0)])
    
dataset['Sex'] = dataset['Sex'].astype('category')
sex_map = cat_map(dataset['Sex'])
dataset['Sex'] = dataset['Sex'].cat.codes

#Preprocessign the dataset by extracting titles from the Name

to_replace_title = {'mme': 'mrs', 'mlle': 'miss', 'ms': 'miss',}
TitleX = ['master', 'miss', 'mr', 'mrs']
#To_Repl = [item for item in All_TitleX if item not in TitleX]
    
def extract_title(col_name):
    t = col_name.apply(lambda Name: re.sub(r'(.*, )|(\..*)', '', Name).lower()).astype(str)
    t = t.replace(to_replace_title)
    t = t.apply(lambda x: 'rare Title' if x not in TitleX else x)
    return t 

dataset['Title'] = extract_title(dataset['Name'])

dataset['Title'] = dataset['Title'].astype("category")
title_map = cat_map(dataset['Title'])
dataset['Title'] = dataset['Title'].cat.codes

featureLinearModel = ['Pclass', 'Sex','SibSp']
ImputColumnLinearModel = ['Age']

from sklearn import linear_model

def lin_mod(input):
    """ Create linear model """
    imput = input[input.Age.notnull()]
    inputFeature = imput[featureLinearModel]
    ImputeColumn = imput[ImputColumnLinearModel]
    lm = linear_model.LinearRegression()
    lm.fit(inputFeature, ImputeColumn)
    return lm

linear_mod = lin_mod(dataset)

#Predict Age and Remove the PredictedAge column once the Age null and NotNull datasets are merged.
    
dataset['PredictedAge'] = linear_mod.predict(dataset[featureLinearModel])
dataset['Age'] = dataset.apply(lambda x: x.Age if pd.notnull(x.Age) else x.PredictedAge, axis=1)
dataset.drop(['PredictedAge'], axis=1, inplace=True)


inp_col = ['Pclass','Sex','Age','SibSp','Parch','Title',]
res_col = 'Survived'

#Building RandomForestRegressor Model for Survived
from sklearn.ensemble import RandomForestClassifier

def ran_for(input):
    input_col = input[inp_col]
    result_col = input[res_col]
    rf = RandomForestClassifier(n_estimators = 500, max_features = 2, random_state = 0 )
    rf.fit(input_col, result_col)
    return rf
    
randForestReg = ran_for(dataset)

from sklearn.externals import joblib
def store_file(val, file):
    joblib.dump(val, file)

store_file(randForestReg, 'random_forest.mdl')
store_file(title_map, 'title_map.file')
store_file(sex_map, 'sex_map.file')