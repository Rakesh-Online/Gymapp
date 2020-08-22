#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 22:01:24 2019

@author: kanth
"""

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


data = pd.read_excel('dataGYM.xlsx')

data.columns

del data['BMI']




X = data.iloc[:,:3]

y = data.iloc[:,3:]


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
data['Prediction'] = label_encoder.fit_transform(data['Prediction'])


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

model_GYM = RandomForestClassifier()



model_GYM = LogisticRegression()
model_GYM.fit(X_train, y_train)

print(model_GYM)
# make predictions
expected = y_test
predicted = model_GYM.predict(X_test)
# summarize the fit of the model
#Correction
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

import pickle

pickle.dump(model, open("model_GYM.pkl", "wb" ))

my_scaler = pickle.load(open("model_GYM.pkl", "rb" ))





