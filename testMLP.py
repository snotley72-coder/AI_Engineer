# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 10:40:56 2024

@author: Scott
"""

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib as JobLib
import numpy as np
import csv
import time
import matplotlib.pyplot as plt


def read_2d(file_path):
    data = []
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            data.append(row)
    return data

# Load a trained model

model = JobLib.load('mlp_regressor_model.joblib')

# Read input file and set-up output array

file_path = 'inps.csv'
inputs = read_2d(file_path)
inputs=np.array(np.float64(inputs))

outputs=np.zeros(136)

# Set initial conditions from input file

inps=inputs[0:1,:]
inps[0,0]=0.5

# Run trained MLP in NARX recurrent set up from initial conditions

for i in range(136):
    pred=model.predict(inps)
    outputs[i]=pred

    # Left shift input queue and add predicted output to the
    # input queue.

    inps[0,2:4]=inputs[i,2:4]
    inps[0,7]=inps[0,6]
    inps[0,6]=inps[0,5]
    inps[0,5]=pred

plt.plot(outputs)

