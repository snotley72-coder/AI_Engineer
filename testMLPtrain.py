# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 12:19:30 2024

@author: Scott
"""

# Importing necessary libraries
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib as JobLib
import numpy as np
import csv
import time

def read_2d(file_path):
    data = []
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            data.append(row)
    return data

# Read in Power, Velocity, Position and prevDepths
# (outputs are set to create a one-step ahead predictor)

file_path = 'inps.csv'
inputs = read_2d(file_path)

file_path = 'outs.csv'
outputs = read_2d(file_path)

inps=np.array(np.float64(inputs))
outs=np.array(np.float64(outputs))

current_time=int(time.time())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(inps, outs, test_size=0.2, random_state=current_time)

# Create an MLP Regressor model
mlp_regressor = MLPRegressor(hidden_layer_sizes=(150), activation='relu', max_iter=32000, random_state=current_time)

# Fit the model to the training data
mlp_regressor.fit(X_train, y_train.ravel())

# Make predictions on the test set
y_pred = mlp_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plotting the results
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, color='black', label='Actual')
plt.title('MLP Regressor Prediction')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()


JobLib.dump(mlp_regressor, 'mlp_regressor_model.joblib')
