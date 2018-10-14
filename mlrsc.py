"""Making Linear Regression from scratch"""

import numpy as np
import pandas as pd

# setting up the costant parameters
alpha = 0.003
np.random.seed(42)
theta = np.random.rand(len(data.columns)).reshape(-1, 1)

# loading data
data = pd.read_csv("data2.txt")

# feature scaling
def feature_scaling(array):
    array = (array - array.mean())/array.std()
    return array

data = feature_scaling(data)

# making feature and target variables
X = data.iloc[:, 0:2].values
y = data.iloc[:, 2].values.reshape(-1, 1)

# append an extra column for theta0
X = np.append(np.ones([X.shape[0], 1]), X, axis = 1)

# cost function to understand how the theta values converge
def cost_prediction(X, y, theta):
    prediction = np.dot(X, theta) - y
    to_sum = np.power(prediction, 2)
    j = np.sum(to_sum)/(2*len(y))
    return j

# gradient descent to find optimal parameters
def gradient_descent(X, y, theta, alpha):
    for i in range(5000):
        prediction = X@theta.T.reshape(-1, 1) - y
        sub_theta = (X.T@prediction)*(alpha/len(y))
        theta -= sub_theta
        cost = cost_prediction(X, y, theta)
        if i%10 == 0:
            print(cost)
    return theta

# to get predicted output
y_pred = X@theta