import numpy as np
import json

# Constants
NUM_FEATURES = 100

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

def predict_perceptron(row, weights):
    activation = np.dot(row, weights)
    return 1.0 if activation >= 0.0 else 0.0

def fit_perceptron(X, y, lr, n_epoch, n):
    p = len(X[0])
    w = np.zeros((p,1))
    for epoch in range(n_epoch):
        # sum_error = 0.0
        for row, actual in zip(X[:n], y[:n]):
            prediction = predict_perceptron(row, w)
            error = actual - prediction
            # sum_error += abs(actual - prediction)
            w += lr * error * np.expand_dims(row, axis=1)
        # print('> epoch=%d, lrate=%.3f, error=%.3f, acc=%.3f' % (epoch, lr, sum_error/len(X), accuracy_perceptron(X[n:], y[n:], w)))
    return w

def accuracy_perceptron(X, y, weights):
    sum_error = 0
    for row, actual in zip(X, y):
        prediction = predict_perceptron(row, weights)
        error = actual - prediction
        sum_error += error**2
    return (len(X) - sum_error) / len(X)

def parameter_accuracy(weights, actual):
    norm_weights = weights / np.linalg.norm(weights)
    norm_actual = actual / np.linalg.norm(actual)
    return np.sqrt(((norm_weights - norm_actual) ** 2).mean())

def accuracy_logistic(X, y, weights):
    sum_error = 0
    for row, actual in zip(X, y):
        pred = sigmoid(np.dot(row, weights)) - 0.5
        prediction = (pred / abs(pred)) / 2 + 0.5
        error = actual - prediction[0]
        sum_error += error**2
    return (len(X) - sum_error) / len(X)

def logistic_loss(y, pred):
    loss = -np.mean(y*(np.log(pred)) + (1-y)*np.log(1-pred))
    return loss

def logistic_gradients(X, y, pred):
    n = X.shape[0]
    dw = (1/n)*np.dot(np.expand_dims(X.T, axis=1), (pred - y))
    return dw

def fit_logistic(X, y, lr, epochs, n):
    total_n, p = X.shape
    
    # Initializing weights and bias to zeros.
    w = np.zeros((p,1))
    
    # Reshaping y.
    y = y.reshape(total_n,1)
    
    # Training loop.
    for epoch in range(epochs):
        for i in range(n):
            
            # Calculating hypothesis/prediction.
            y_hat = sigmoid(np.dot(X[i], w))
            # Getting the gradients of loss w.r.t parameters.
            dw = np.expand_dims(logistic_gradients(X[i], y[i], y_hat), axis=1)
            # Updating the parameters.
            w -= lr*dw
        
        # Calculating loss and appending it in the list.
        # l = logistic_loss(y, sigmoid(np.dot(X, w)))
        # print('> epoch=%d, lrate=%.3f, error=%.3f, acc=%.3f' % (epoch, lr, l, accuracy_logistic(X[n:], y[n:], w)))
    # returning weights, bias and losses(List).
    return w

def run_beta(n, max_beta, seed=0):
    res = []
    for beta in range(1, max_beta):
        theta, X, y = generate_logit_data(beta, n, seed)
        p_weights = fit_perceptron(X, y, 0.1, 25, n)
        l_weights = fit_logistic(X, y, 0.1, 25, n)

        p_acc = parameter_accuracy(p_weights, theta)
        l_acc = parameter_accuracy(l_weights, theta)
        print(f'Perceptron accuracy (n={n}, b={beta}): {p_acc}')
        print(f'Logistic accuracy (n={n}, b={beta}): {l_acc}')
        
        res.append((n, beta, p_acc, l_acc))
    return res

def run_n(beta, max_n, seed=0):
    res = []
    for n in range(1, max_n, 500):
        theta, X, y = generate_logit_data(beta, n, seed)
        p_weights = fit_perceptron(X, y, 0.1, 25, n)
        l_weights = fit_logistic(X, y, 0.1, 25, n)

        p_acc = parameter_accuracy(p_weights, theta)
        l_acc = parameter_accuracy(l_weights, theta)
        print(f'Perceptron accuracy (n={n}, b={beta}): {p_acc}')
        print(f'Logistic accuracy (n={n}, b={beta}): {l_acc}')
        
        res.append((n, beta, p_acc, l_acc))
    return res

if __name__ == '__main__':
    iter = 25
    res = run_beta(25000, 40, iter)
    with open(f'vary_beta/{iter}.txt', 'w') as f:
        f.write(json.dumps(res))
    res = run_n(5, 25000, iter)
    with open(f'vary_n/{iter}.txt', 'w') as f:
        f.write(json.dumps(res))
    