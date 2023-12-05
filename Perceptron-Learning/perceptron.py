import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
import json
import sys

# Constants
NUM_FEATURES = 2
EPSILON = 0.5

# Calculate the sigmoid / logistic function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Dataset generation
#   Generate an array of floats from 0 to 1, and dot with theta to find z
#   Probabilities result from applying sigmoid to the z array
#   Theta is drawn from N(0,1) and normalized to a given value of beta
#   X is drawn randomly from U[-1, 1]
def generate_data(beta, n, seed):
    np.random.seed(seed)
    theta_gen = np.random.normal(size=(NUM_FEATURES, 1))
    theta_gen = theta_gen * beta / np.linalg.norm(theta_gen)
    X_gen = np.random.normal(0, 1, (n, NUM_FEATURES))
    z_gen = np.dot(X_gen, theta_gen)
    y_gen = np.random.binomial(1, sigmoid(z_gen))
    return theta_gen, X_gen, np.ndarray.flatten(y_gen)

if __name__ == '__main__':
    beta = 50
    theta, X, y = generate_data(beta, 20000, 1)
    model = Perceptron(fit_intercept=False)
    model.fit(X[:10000], y[:10000])

    logit_model = LogisticRegression(fit_intercept=False, penalty=None)
    logit_model.fit(X[:10000], y[:10000])

    print(model.score(X[10000:], y[10000:]))
    print(logit_model.score(X[10000:], y[10000:]))

    print(np.linalg.norm(model.coef_))
    print(np.linalg.norm(logit_model.coef_))
    # direction_error = np.linalg.norm(theta / beta - model.coef_ / np.linalg.norm(model.coef_))
    # magnitude_error = (beta - np.linalg.norm(model.coef_)) / beta