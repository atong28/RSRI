import numpy as np
import cvxpy as cp
from sklearn.model_selection import train_test_split
import json
import sys

# Constants
NUM_FEATURES = 100
NUM_TRAIN_SAMPLES = 0
EPSILON = 0.5

# Calculate the sigmoid / logistic function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Dataset generation
#   Generate an array of floats from 0 to 1, and dot with theta to find z
#   Probabilities result from applying sigmoid to the z array
#   Theta is drawn from N(0,1) and normalized to a given value of beta
#   X is drawn randomly from U[-1, 1]
def generate_data(beta, n):
    np.random.seed(SEED)
    theta_gen = np.random.normal(size=(NUM_FEATURES, 1))
    theta_gen = theta_gen * beta / np.linalg.norm(theta_gen)
    X_gen = np.random.rand(n, NUM_FEATURES) * 2 - 1
    z_gen = np.dot(X_gen, theta_gen)
    y_gen = np.random.binomial(1, sigmoid(z_gen))
    return theta_gen, X_gen, y_gen
    
# Logistic Regression Object
class LogisticRegression:
    def __init__(self, X, y, beta, theta, n):
        self.X = X
        self.y = y
        self.beta = beta
        self.theta = np.ndarray.flatten(theta)
        self.n = n
        self.problem = self.setup()

    # Initialize the CVXPY maximization problem.
    def setup(self):
        self.weights = cp.Variable(NUM_FEATURES)
        log_likelihood = cp.sum(
            cp.multiply(np.ndarray.flatten(self.y), self.X @ self.weights) - cp.logistic(self.X @ self.weights)
        )
        constraints = [cp.norm(self.weights) <= self.beta]
        return cp.Problem(cp.Maximize(log_likelihood/self.n), constraints)
    
    # Solve the problem, and calculate various measurements for performance
    def run(self):
        self.problem.solve(solver=cp.CLARABEL)
        self.RMSE = np.linalg.norm(self.theta - self.weights.value) / self.beta
        self.RMSE2 = np.linalg.norm(self.theta / self.beta - self.weights.value / cp.norm(self.weights).value)
        if abs(np.linalg.norm(self.theta) - cp.norm(self.weights).value) > EPSILON:
            print(f'Solve status: {self.problem.status}')
            print(f'Theta: {self.theta}')
            print(f'Weights: {self.weights.value}')
        print(f'Beta: {self.beta} | n: {self.n} | RMSE: {self.RMSE} | RMSE2: {self.RMSE2} | Magnitude of weights: {cp.norm(self.weights).value}')

def run(beta, n):
    theta, X, y = generate_data(beta, n)

    regressor = LogisticRegression(X, y, beta, theta, n)
    regressor.run()
    return regressor

def vary_beta():
    print(f'Initializing with seed {SEED}, n={NUM_TRAIN_SAMPLES}.')
    with open(f'vary_beta/data_{SEED}.json', 'r') as f: 
        data = json.load(f)
    try:
        for beta in range(1, 101):
            try:
                regressor = run(beta, NUM_TRAIN_SAMPLES)
                data[str(NUM_TRAIN_SAMPLES)][beta].append(regressor.RMSE)
            except KeyboardInterrupt:
                print('Interrupted')
                sys.exit(0)
            except:
                print(f'Error reached at beta = {beta}')
                continue
    except KeyboardInterrupt:
        print('Interrupted')
        sys.exit(0)
    finally:
        with open(f'vary_beta/data_{SEED}.json', 'w') as f: 
            f.write(json.dumps(data))

def vary_n():
    print(f'Initializing with seed {SEED}, beta={BETA}.')
    with open(f'vary_n/data_{SEED}.json', 'r') as f: 
        data = json.load(f)
    try:
        for n in range(500, 25001, 500):
            try:
                regressor = run(BETA, n)
                data[str(BETA)][n//500].append(regressor.RMSE)
            except KeyboardInterrupt:
                print('Interrupted')
                sys.exit(0)
            except:
                print(f'Error reached at beta = {n}')
                continue
    except KeyboardInterrupt:
        print('Interrupted')
        sys.exit(0)
    finally:
        with open(f'vary_n/data_{SEED}.json', 'w') as f: 
            f.write(json.dumps(data))

# Run with arguments: python3 cvxpy-beta.py <mode> <seed> <n>
# Mode 1: vary beta, fixed n with output RMSE
# Mode 2: vary n, fixed beta with output RMSE
if __name__ == '__main__':
    match int(sys.argv[1]):
        case 1:
            SEED = int(sys.argv[2])
            NUM_TRAIN_SAMPLES = int(sys.argv[3])
            vary_beta()
        case 2:
            SEED = int(sys.argv[2])
            BETA = float(sys.argv[3])
            vary_n()