import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import math

np.random.seed(3)

# Constants
learning_rate = 0.0001
num_iters = 4000
bias = False

# Calculate the sigmoid / logistic function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 0-1 accuracy model
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

# Dataset generation
#   p: number of features
#   m: norm of the parameter vector
#   theta: parameter vector for generation
#   n: number of samples in training set
#   t: number of samples in testing set
#   Generate an n x p array of floats from 0 to 1, and dot with theta to find z
#   Probabilities result from applying sigmoid to the z array
p = 100
m = 30
theta = np.random.random(size=(p, 1)) * 4 - 2
theta = theta * m / np.linalg.norm(theta)

print(f'Norm of parameter vector: {np.linalg.norm(theta)}')
n = 25000
t = 500
X = np.random.rand(n+t, p)
z = np.dot(X, theta)
prob = sigmoid(z)

# Generate labels by sampling from Bernoulli(prob)
y = np.random.binomial(1, prob.flatten())

# Generate labels deterministically
## y = np.where(prob.flatten() >= 0.5, 1, 0)

# Split data into training and testing data
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=t/(n+t), random_state=0)

# Train the model
iter_list = []
rms_list = []

# plot as n increases the RMSE
for iter in range(100, 25000, 100):
    regressor = LogisticRegression()
    regressor.fit(xtrain[:iter], ytrain[:iter])
    print(f'Trained weights with n={iter} has norm {np.linalg.norm(regressor.coef_)} with accuracy {accuracy(ytest, regressor.predict(xtest))}')
    
    ## Formula 1
    MSE = np.linalg.norm(np.subtract(theta,regressor.coef_.T)) / np.linalg.norm(theta)
    
    ## Formula 2
    # MSE = np.linalg.norm(np.subtract(theta / np.linalg.norm(theta),regressor.coef_.T / np.linalg.norm(regressor.coef_.T)))
    
    RMSE = MSE / math.sqrt(p)
    print(f'RMSE: {RMSE}')
    
    iter_list.append(iter)
    rms_list.append(RMSE)
    
plt.plot(iter_list, rms_list)
plt.xlabel("Number of iteration")
plt.ylabel("RMSE")
plt.show()