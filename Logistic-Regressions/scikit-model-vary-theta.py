import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import math

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

iter_list = []
rms_list = []
diff_list = []

for m in range(1, 80):
    np.random.seed(3)

    # Dataset generation
    #   p: number of features
    #   m: norm of the parameter vector
    #   theta: parameter vector for generation
    #   n: number of samples in training set
    #   t: number of samples in testing set
    #   Generate an n x p array of floats from 0 to 1, and dot with theta to find z
    #   Probabilities result from applying sigmoid to the z array
    p = 100
    theta = np.random.random(size=(p, 1)) * 4 - 2
    theta = theta * m / np.linalg.norm(theta)

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
    

    regressor = LogisticRegression(max_iter=10000)
    regressor.fit(xtrain, ytrain)
    print(f'Trained weights with norm of theta={m} has norm {np.linalg.norm(regressor.coef_)} (diff {m - np.linalg.norm(regressor.coef_)}) with accuracy {accuracy(ytest, regressor.predict(xtest))}')
    ## Formula 1
    # MSE = np.linalg.norm(np.subtract(theta,regressor.coef_.T)) / np.linalg.norm(theta)
    
    ## Formula 2
    MSE = np.linalg.norm(np.subtract(theta / np.linalg.norm(theta),regressor.coef_.T / np.linalg.norm(regressor.coef_.T)))
    RMSE = MSE / math.sqrt(p)
    print(f'RMSE: {RMSE}')
    
    iter_list.append(m)
    rms_list.append(RMSE)
    diff_list.append(m - np.linalg.norm(regressor.coef_))
        
plt.plot(iter_list, rms_list)
plt.xlabel("Norm of theta")
plt.ylabel("RMSE")
plt.show()