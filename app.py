# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 17:06:01 2019

@author: prithvi
"""
import flask
from flask import Flask, request , jsonify, render_template
#import jinja2
import numpy as np
import pandas as pd
import pickle

model = open("model_GYM.pkl", "rb")
model = pickle.load(model)

app = Flask(__name__, template_folder='templates')

@app.route("/",methods=["GET"])
def index():
    return render_template('home.html')

@app.route("/", methods=["POST"])
def preprocess():

    name = request.form['input']
    x = list(name.split(","))
    x, y, z = (i for i in x)
    simple_list = [[x, y, z]]
    df=pd.DataFrame(simple_list,columns=['x', 'y', 'z'])
    X = df.iloc[:, :].values
    X = X[0]
    X = np.reshape(X, (1, -1))
    model_o = model.predict(X)
    model_o[0]
    meow = [ (x.split("--")) for x in model_o]
    if meow[0][0] != 'Healthy':
        zwee = [x.split("->") for x in meow[0]]
        o_1 = zwee[0][0]
        o_2 = zwee[1][0]
        o_3 = zwee[1][1]
        line_1 = "You are " + o_1
        line_2 = "The following is the plan for you to become {} : {}".format(o_3, o_2)
    else:
        o_1 = meow[0][0]
        o_2 = meow[0][1]
        line_1 = "You are " + o_1
        line_2 = "The following is the plan for you : {}".format(o_2)
    return render_template('home.html', line_1= line_1, line_2 = line_2)

#class DataForm(Form):

if __name__ == "__main__":
    app.run()
