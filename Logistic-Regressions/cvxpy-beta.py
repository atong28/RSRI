import numpy as np
import cvxpy as cp
from sklearn.model_selection import train_test_split
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
def generate_data(beta, n, seed):
    np.random.seed(seed)
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
        return cp.Problem(cp.Maximize(log_likelihood), constraints)
    
    # Solve the problem, and calculate various measurements for performance
    def run(self):
        self.problem.solve(solver=cp.CLARABEL)
        self.RMSE = np.linalg.norm(self.theta - self.weights.value) / self.beta
        if abs(np.linalg.norm(self.theta) - cp.norm(self.weights).value) > EPSILON:
            print(f'Solve status: {self.problem.status}')
            print(f'Theta: {self.theta}')
            print(f'Weights: {self.weights.value}')
        print(f'Beta: {self.beta} | n: {self.n} | RMSE: {self.RMSE} | Magnitude of weights: {cp.norm(self.weights).value}')

def run(beta, n, seed):
    theta, X, y = generate_data(beta, n, seed)

    regressor = LogisticRegression(X, y, beta, theta, n)
    regressor.run()
    return regressor

def vary_beta(seed, n):
    print(f'Initializing with seed {seed}, n={n}.')
    with open(f'vary_beta/data_{seed}.json', 'r') as f: 
        data = json.load(f)
    try:
        for beta in range(1, 101):
            try:
                regressor = run(beta, n, seed)
                data[str(n)][beta].append(regressor.RMSE)
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
        with open(f'vary_beta/data_{seed}.json', 'w') as f: 
            f.write(json.dumps(data))

def vary_n(seed, beta):
    print(f'Initializing with seed {seed}, beta={beta}.')
    with open(f'vary_n/data_{seed}.json', 'r') as f: 
        data = json.load(f)
    try:
        for n in range(500, 25001, 500):
            try:
                regressor = run(beta, n, seed)
                data[str(beta)][n//500] = [regressor.RMSE]
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
        with open(f'vary_n/data_{seed}.json', 'w') as f: 
            f.write(json.dumps(data))

def tighten_estimation(n, beta, delta, iters, step):
    print(f'Initializing with n=[{n-delta},{n+delta}], beta={beta}, step size={step}, iters={iters}')
    try:
        with open(f'vary_RMSE/data_{beta}.json', 'r') as f:
            data = json.load(f)
    except:
        data = {}

    try:
        for sample_size in range(n-delta, n+delta+1, step):
            buffer_seed = 1
            if str(sample_size) in data:
                rmse_list = data[str(sample_size)]
                buffer_seed += len(rmse_list)
                print(f'Processed pre-existing data for n={sample_size}')
            else:
                rmse_list = []
            for seed in range(buffer_seed, iters+buffer_seed):
                try:
                    print(f'Seed: {seed} | ', end="")
                    regressor = run(beta, sample_size, seed)
                    rmse_list.append(regressor.RMSE)
                except KeyboardInterrupt:
                    print('Interrupted')
                    sys.exit(0)
                except:
                    print(f'Error reached at beta = {n}')
                    continue
            data[str(sample_size)] = rmse_list
    except KeyboardInterrupt:
        print('Interrupted')
        sys.exit(0)
    finally:
        with open(f'vary_RMSE/data_{beta}.json', 'w') as f:
            f.write(json.dumps(data))

# Run with arguments: python3 cvxpy-beta.py <mode> <seed> <n>
# Mode 1: vary beta, fixed n with output RMSE
# Mode 2: vary n, fixed beta with output RMSE
# Mode 3: vary n, fixed beta at a target RMSE, n = center, beta = beta, delta = spread, iters = how many repetitions, step = distances between each n tested
if __name__ == '__main__':
    match int(sys.argv[1]):
        case 1:
            seed = int(sys.argv[2])
            n = int(sys.argv[3])
            vary_beta(seed, n)
        case 2:
            seed = int(sys.argv[2])
            beta = float(sys.argv[3])
            vary_n(seed, beta)
        case 3:
            n = int(sys.argv[2])
            beta = float(sys.argv[3])
            delta = int(sys.argv[4])
            iters = int(sys.argv[5])
            step = int(sys.argv[6])
            tighten_estimation(n, beta, delta, iters, step)