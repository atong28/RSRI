import numpy as np
from sklearn.model_selection import train_test_split
from numpy import log,dot,e,shape
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

# Calculates the cost function (logistic loss)
def cost(X, y, theta):
    z = dot(X,theta)
    cost0 = y.T.dot(log(sigmoid(z)))
    cost1 = (1-y).T.dot(log(1-sigmoid(z)))
    cost = -((cost1 + cost0))/len(y) 
    return cost



# Dataset generation
#   p: number of features
#   m: norm of the parameter vector
#   theta: parameter vector for generation
#   n: number of samples in training set
#   t: number of samples in testing set
#   Generate an n x p array of floats from 0 to 1, and dot with theta to find z
#   Probabilities result from applying sigmoid to the z array
p = 100
m = 100
theta = np.random.random(size=(p, 1)) * 4 - 2
theta = theta * m / np.linalg.norm(theta)

print(f'Norm of parameter vector: {np.linalg.norm(theta)}')
n =20000
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
    
class LogisticRegression:

    def __init__(self, alpha=0.0001, iters=3000):
        self.alpha = alpha
        self.iters = iters
        self.weights = None
        self.X = None

    # Initialize weights and test data to include bias factor
    def initialize(self, X):
        weights = np.zeros((shape(X)[1]+1,1))
        X = np.c_[np.ones((shape(X)[0],1)), X]
        return weights, X

    def fit(self, X, y):
        if bias: weights, X = self.initialize(X)
        else: weights = np.zeros((shape(X)[1], 1))
        cost_list = np.zeros(self.iters,)
        for i in range(self.iters):
            weights = weights - self.alpha * dot(X.T, sigmoid(dot(X,weights)) - np.reshape(y,(len(y),1)))
            cost_list[i] = cost(X, y, weights)
        self.weights = weights
        return cost_list
    
    def predict(self, X):
        if bias: z = dot(self.initialize(X)[1],self.weights)
        else: z = dot(X, self.weights)
        l = [1 if i > 0.5 else 0 for i in sigmoid(z)]
        return np.array(l)
    
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

iter_list = []
rms_list = []

# plot as n increases the RMSE
for iter in range(500, 20000, 500):
    regressor = LogisticRegression(alpha = learning_rate, iters = num_iters)
    regressor.fit(xtrain[:iter], ytrain[:iter])
    print(f'Trained weights with n={iter} has norm {np.linalg.norm(regressor.weights)} with accuracy {accuracy(ytest, regressor.predict(xtest))}')
    # MSE = np.linalg.norm(np.subtract(theta / np.linalg.norm(theta),regressor.weights.T / np.linalg.norm(regressor.weights.T)))
    MSE = np.linalg.norm(np.subtract(theta,regressor.weights.T))
    RMSE = MSE / math.sqrt(p)
    print(f'RMSE: {RMSE}')
    
    iter_list.append(iter)
    rms_list.append(RMSE)
    
plt.scatter(iter_list, rms_list,color="r")
plt.plot(iter_list, rms_list)
plt.xlabel("Number of iteration")
plt.ylabel("RMSE")
plt.show()