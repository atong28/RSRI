import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import json
import sys

# Constants
NUM_FEATURES = 100
EPSILON = 0.5

# Calculate the sigmoid / logistic function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Dataset generation
#   Generate an array of floats from 0 to 1, and dot with theta to find z
#   Probabilities result from applying sigmoid to the z array
#   Theta is drawn from N(0,1) and normalized to a given value of beta
#   X is drawn randomly from U[-1, 1]
def generate_logit_data(beta, n, seed):
    np.random.seed(seed)
    theta_gen = np.random.normal(size=(NUM_FEATURES, 1))
    theta_gen = theta_gen * beta / np.linalg.norm(theta_gen)
    X_gen = np.random.normal(0, 1, (n, NUM_FEATURES))
    z_gen = np.dot(X_gen, theta_gen)
    y_gen = np.random.binomial(1, sigmoid(z_gen))
    return theta_gen, X_gen, np.ndarray.flatten(y_gen)

def generate_probit_data(beta, n, seed):
    np.random.seed(seed)
    theta_gen = np.random.normal(size=(NUM_FEATURES, 1))
    theta_gen = theta_gen * beta / np.linalg.norm(theta_gen)
    X_gen = np.random.normal(0, 1, (n, NUM_FEATURES))
    z_gen = np.dot(X_gen, theta_gen)
    y_gen = (np.sign(z_gen + np.random.normal(0, 1, (np.shape(z_gen)))) + 1) / 2
    return theta_gen, X_gen, np.ndarray.flatten(y_gen)

if __name__ == '__main__':
    '''
    n = 10000
    

    beta_values = list(range(1, 40))
    p_scores = []
    l_scores = []

    for beta in beta_values:
        theta, X, y = generate_logit_data(beta, n+5000, 1)
        probit_model = Perceptron(fit_intercept=False)
        probit_model.fit(X[:n], y[:n])

        logit_model = LogisticRegression(fit_intercept=False, penalty=None)
        logit_model.fit(X[:n], y[:n])

        p_scores.append(probit_model.score(X[n:], y[n:]))
        l_scores.append(logit_model.score(X[n:], y[n:]))

    pline, = plt.plot(beta_values, p_scores)
    lline, = plt.plot(beta_values, l_scores)

    plt.xlabel("n")
    plt.ylabel("Accuracy")
    plt.legend([lline, pline], ["Logit model", "Probit model"])
    plt.show()
    '''
    #'''
    beta = 40
    theta, X, y = generate_logit_data(beta, 30000, 1)

    n_values = list(range(1000, 25000, 250))
    p_scores = []
    l_scores = []

    for n in n_values:

        probit_model = Perceptron(fit_intercept=False)
        probit_model.fit(X[:n], y[:n])

        logit_model = LogisticRegression(fit_intercept=False, penalty=None, max_iter=1000)
        logit_model.fit(X[:n], y[:n])

        p_scores.append(probit_model.score(X[25000:], y[25000:]))
        l_scores.append(logit_model.score(X[25000:], y[25000:]))

    pline, = plt.plot(n_values, p_scores)
    lline, = plt.plot(n_values, l_scores)

    plt.xlabel("n")
    plt.ylabel("Accuracy")
    plt.legend([lline, pline], ["Logit model", "Probit model"])
    plt.show()
    #'''


    # dir_error = np.linalg.norm(model.coef_/np.linalg.norm(model.coef_) - logit_model.coef_/np.linalg.norm(logit_model.coef_))
    # len_error = np.linalg.norm(model.coef_) - np.linalg.norm(logit_model.coef_)/1.65 

    # print(f'Directional error: {dir_error}')
    # print(f'Length error: {len_error}')

    # print(f'Real: {np.linalg.norm(theta)} | Probit*1.65: {np.linalg.norm(model.coef_) * 1.65} | Logit: {np.linalg.norm(logit_model.coef_)}')
    # direction_error = np.linalg.norm(theta / beta - model.coef_ / np.linalg.norm(model.coef_))
    # magnitude_error = (beta - np.linalg.norm(model.coef_)) / beta