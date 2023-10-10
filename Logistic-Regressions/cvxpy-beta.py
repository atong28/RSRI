import numpy as np
import cvxpy as cp
from sklearn.model_selection import train_test_split
from numpy import shape

np.random.seed(1)

# Constants
NUM_FEATURES = 100
NUM_TRAIN_SAMPLES = 25000
NUM_TEST_SAMPLES = 500
NUM_TOTAL_SAMPLES = NUM_TRAIN_SAMPLES + NUM_TEST_SAMPLES
BETA = 1
LAMBDA = 0.5
EPSILON = 0

# Calculate the sigmoid / logistic function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Dataset generation
#   Generate an array of floats from 0 to 1, and dot with theta to find z
#   Probabilities result from applying sigmoid to the z array
def generate_data():
    theta_gen = np.random.random(size=(NUM_FEATURES, 1)) * 2 - 1
    theta_gen = theta_gen * BETA / np.linalg.norm(theta_gen)
    X_gen = np.random.rand(NUM_TOTAL_SAMPLES, NUM_FEATURES)
    z_gen = np.dot(X_gen, theta_gen)
    y_gen = np.random.binomial(1, sigmoid(z_gen))
    return theta_gen, X_gen, y_gen
    
class LogisticRegression:

    def __init__(self, xtrain, ytrain, xtest, ytest, theta):
        self.X = xtrain
        self.y = ytrain
        self.xtest = xtest
        self.ytest = np.ndarray.flatten(ytest)
        self.theta = np.ndarray.flatten(theta)
        self.problem = self.setup()

    def setup(self):
        self.weights = cp.Variable(NUM_FEATURES)
        log_likelihood = cp.sum(
            cp.multiply(np.ndarray.flatten(self.y), self.X @ self.weights) - cp.logistic(self.X @ self.weights)
        )
        constraints = [cp.norm(self.weights) <= BETA + EPSILON]
        return cp.Problem(cp.Maximize(log_likelihood/NUM_TRAIN_SAMPLES), constraints)
    
    def run(self):
        self.problem.solve()
        self.RMSE = np.linalg.norm(self.theta - self.weights.value) / (np.sqrt(NUM_FEATURES) * BETA)
        print(f'Solve status: {self.problem.status}')
        print(f'Magnitude of theta: {np.linalg.norm(self.theta)}, Shape of X {shape(self.X)}, shape of y {shape(self.y)}')
        print(f'Magnitude of weights: {cp.norm(self.weights).value}')
        print(f'RMSE: {self.RMSE}')
        test_error = self.predict()

        return test_error
    
    def predict(self):
        z = (self.xtest @ self.weights.value)
        l = np.array([1 if i > 0.5 else 0 for i in sigmoid(z)])
        return np.sum(self.ytest != l) / len(self.ytest)

def run():
    theta, X, y = generate_data()

    # Split data into training and testing data
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=NUM_TEST_SAMPLES/(NUM_TOTAL_SAMPLES), random_state=0)

    regressor = LogisticRegression(xtrain, ytrain, xtest, ytest, theta)
    test_error = regressor.run()
    print(f'Test error: {test_error}')
    
run()