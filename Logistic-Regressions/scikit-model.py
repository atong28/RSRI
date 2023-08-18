import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

np.random.seed(3)

# Calculate the sigmoid / logistic function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Dataset generation
#   theta: parameter vector for generation
#   p: number of features
#   n: number of samples to generate
#   Generate an n x p array of floats from 0 to 1, and dot with theta to find z
#   Probabilities result from applying sigmoid to the z array
theta = np.random.random(size=(100, 1)) * 2 - 1
p = len(theta)
n = 2000
X = np.random.rand(n, p)
z = np.dot(X, theta)
prob = sigmoid(z)

# Generate labels by sampling from Bernoulli(prob)
y = np.random.binomial(1, prob.flatten())

# Generate labels deterministically
## y = np.where(prob.flatten() >= 0.5, 1, 0)

# Split data into training and testing data
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)

# 0-1 accuracy model
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

# Train the model
regressor = LogisticRegression(fit_intercept = False, penalty = None)
regressor.fit(xtrain, ytrain)
print(f'Trained weights: {regressor.coef_} has norm {np.linalg.norm(regressor.coef_)} with accuracy {accuracy(ytest, regressor.predict(xtest))}')