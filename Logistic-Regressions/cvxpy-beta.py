import numpy as np
import cvxpy as cp
from sklearn.model_selection import train_test_split
import json
import sys

# Constants
SEED = 1
NUM_FEATURES = 100
NUM_TRAIN_SAMPLES = 10000
NUM_TEST_SAMPLES = 1
NUM_TOTAL_SAMPLES = NUM_TRAIN_SAMPLES + NUM_TEST_SAMPLES
EPSILON = 0.5

# Calculate the sigmoid / logistic function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Dataset generation
#   Generate an array of floats from 0 to 1, and dot with theta to find z
#   Probabilities result from applying sigmoid to the z array
#   Theta is drawn from N(0,1) and normalized to a given value of beta
#   X is drawn randomly from U[-1, 1]
def generate_data(beta):
    np.random.seed(SEED)
    theta_gen = np.random.normal(size=(NUM_FEATURES, 1))
    theta_gen = theta_gen * beta / np.linalg.norm(theta_gen)
    X_gen = np.random.rand(NUM_TOTAL_SAMPLES, NUM_FEATURES) * 2 - 1
    z_gen = np.dot(X_gen, theta_gen)
    y_gen = np.random.binomial(1, sigmoid(z_gen))
    return theta_gen, X_gen, y_gen
    
# Logistic Regression Object
class LogisticRegression:
    def __init__(self, xtrain, ytrain, beta, xtest, ytest, theta):
        self.X = xtrain
        self.y = ytrain
        self.beta = beta
        self.xtest = xtest
        self.ytest = np.ndarray.flatten(ytest)
        self.theta = np.ndarray.flatten(theta)
        self.problem = self.setup()

    # Initialize the CVXPY maximization problem.
    def setup(self):
        self.weights = cp.Variable(NUM_FEATURES)
        log_likelihood = cp.sum(
            cp.multiply(np.ndarray.flatten(self.y), self.X @ self.weights) - cp.logistic(self.X @ self.weights)
        )
        constraints = [cp.norm(self.weights) <= self.beta]
        return cp.Problem(cp.Maximize(log_likelihood/NUM_TRAIN_SAMPLES), constraints)
    
    # Solve the problem, and calculate various measurements for performance
    def run(self):
        self.problem.solve(max_iters=1000)
        self.RMSE = np.linalg.norm(self.theta - self.weights.value) / (np.sqrt(NUM_FEATURES) * self.beta)
        self.RMSE2 = np.linalg.norm(self.theta / self.beta - self.weights.value / cp.norm(self.weights).value) / (np.sqrt(NUM_FEATURES))
        if abs(np.linalg.norm(self.theta) - cp.norm(self.weights).value) > EPSILON:
            print(f'Solve status: {self.problem.status}')
            print(f'Theta: {self.theta}')
            print(f'Weights: {self.weights.value}')
        print(f'Beta: {self.beta} | RMSE: {self.RMSE} | RMSE2: {self.RMSE2} | Magnitude of weights: {cp.norm(self.weights).value}')
    
    def predict(self):
        z = (self.xtest @ self.weights.value)
        l = np.array([1 if i > 0.5 else 0 for i in sigmoid(z)])
        return np.sum(self.ytest != l) / len(self.ytest)

def run(beta):
    theta, X, y = generate_data(beta)

    # Split data into training and testing data
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=NUM_TEST_SAMPLES/(NUM_TOTAL_SAMPLES), random_state=0)

    regressor = LogisticRegression(xtrain, ytrain, beta, xtest, ytest, theta)
    regressor.run()
    return regressor

# Run with arguments: python3 cvxpy-beta.py <seed> <n>
if __name__ == '__main__':
    SEED = int(sys.argv[1])
    NUM_TRAIN_SAMPLES = int(sys.argv[2])
    print(f'Initializing with seed {SEED}, n={NUM_TRAIN_SAMPLES}.')
    with open('data.json', 'r') as f:
        data = json.load(f)
    try:
        for beta in range(1, 101):
            try:
                regressor = run(beta)
                data[str(NUM_TRAIN_SAMPLES)][beta].append(regressor.RMSE)
            except cp.SolverError:
                print(f'Error reached at beta = {beta}')
                continue
    except KeyboardInterrupt:
        pass
    finally:
        with open('data.json', 'w') as f:
            f.write(json.dumps(data))
