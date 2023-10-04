import numpy as np
from sklearn.model_selection import train_test_split
from numpy import log,dot,e,shape
import matplotlib.pyplot as plt
import math

np.random.seed(3)

# Constants
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

def grad(X, y, theta):
    dim = shape(X)
    gradients = np.zeros(dim[1])

    for k in range(dim[1]):
        x_col = X[:,k]
        # s = 0
        gradients[k] = sum(x_col[i] * (-y[i]) * sigmoid(y[i] * x_col[i] * theta[k]) for i in range(dim[0]))
        # for i in range(dim[0]):
        #     s += x_col[i] * (-y[i]) * sigmoid(y[i] * x_col[i] * theta[k])
        # gradients[k] = s / dim[0]
    return gradients

# Dataset generation
#   p: number of features
#   m: norm of the parameter vector
#   theta: parameter vector for generation
#   n: number of samples in training set
#   t: number of samples in testing set
#   Generate an n x p array of floats from 0 to 1, and dot with theta to find z
#   Probabilities result from applying sigmoid to the z array
p = 2
m = 5
theta_gen = np.random.random(size=(p, 1)) * 4 - 2
theta_gen = theta_gen * m / np.linalg.norm(theta_gen)

print(f'Norm of parameter vector: {np.linalg.norm(theta_gen)}')
n = 20000
t = 500
X_gen = np.random.rand(n+t, p)
z = np.dot(X_gen, theta_gen)
prob = sigmoid(z)

# Generate labels by sampling from Bernoulli(prob)
y_gen = np.random.binomial(1, prob.flatten()) * 2 - 1

# Generate labels deterministically
## y = np.where(prob.flatten() >= 0.5, 1, 0)

# Split data into training and testing data
xtrain, xtest, ytrain, ytest = train_test_split(X_gen, y_gen, test_size=t/(n+t), random_state=0)
    
class LogisticRegression:

    def __init__(self, X, y, alpha=0.001, iters=300):
        self.alpha = alpha
        self.iters = iters
        self.weights = None
        self.X = X
        self.y = y

    def fit(self):
        weights = np.zeros((shape(self.X)[1],))
        for _ in range(self.iters):
            weights = weights + self.alpha * grad(self.X, self.y, weights)
        self.weights = weights
    
    def predict(self, xtest):
        z = dot(xtest, self.weights)
        l = [1 if i > 0.5 else -1 for i in sigmoid(z)]
        return np.array(l)
    
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

iter_list = []
rms_list = []

# plot as n increases the RMSE
for iter in range(500, 15000, 500):
    regressor = LogisticRegression(xtrain[:iter], ytrain[:iter])
    regressor.fit()
    print(f'Trained weights with n={iter} has norm {np.linalg.norm(regressor.weights)} with accuracy {accuracy(ytest, regressor.predict(xtest))}')
    # MSE = np.linalg.norm(np.subtract(theta / np.linalg.norm(theta),regressor.weights.T / np.linalg.norm(regressor.weights.T)))
    MSE = np.linalg.norm(np.subtract(theta_gen,regressor.weights.T))
    RMSE = MSE / math.sqrt(p)
    print(f'RMSE: {RMSE}')
    
    iter_list.append(iter)
    rms_list.append(RMSE)
    
plt.scatter(iter_list, rms_list,color="r")
plt.plot(iter_list, rms_list)
plt.xlabel("Number of iteration")
plt.ylabel("RMSE")
plt.show()