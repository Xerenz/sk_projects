import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metric import mean_squared_error, r2_score
from sklearn import datasets
%matplotlib inline

# load dataset
diabetes = datasets.load_diabetes()

# split into X and y
X = diabetes.data[:, np.newaxis, 2]
y = diabetes.target[:]

# split dataset randomly
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, random_state = 2)

# initialize model
model = LinearRegression()

# fit trainig set into the model
model.fit(X_train, y_train)
# predict values
prediction = model.predict(X_val)
# get mse 
mes = mean_squared_error(prediction, y_val)
# get variance
variance = r2_score(prediction, y_val)

print('MES : ', mes)
print('Variance :' variance)

# plot graph
# plot value points
plt.scatter(X_val, y_val, color = 'black')
# plot the straight line
plt.plot(X_val, prediction, color = 'red')

plt.xticks()
plt.yticks()

plt.show()