# My Regents Scholar Research Initiative Project

I am working with Professor Arya Mazumdar to study certain behaviors of logistic regression. Below is the documentation of my experiments and process.

## Motivation

On the sample complexity of estimation in logistic regression | Daniel Hsu, Arya Mazumdar

> The logistic regression model is one of the most popular data generation model in noisy binary classification problems. In this work, we study the sample complexity of estimating the parameters of the logistic regression model up to a given ℓ2 error, in terms of the dimension and the inverse temperature, with standard normal covariates. The inverse temperature controls the signal-to-noise ratio of the data generation process. While both generalization bounds and asymptotic performance of the maximum-likelihood estimator for logistic regression are well-studied, the non-asymptotic sample complexity that shows the dependence on error and the inverse temperature for parameter estimation is absent from previous analyses. We show that the sample complexity curve has two change-points (or critical points) in terms of the inverse temperature, clearly separating the low, moderate, and high temperature regimes.

[Link to the paper here.](https://arxiv.org/pdf/2307.04191.pdf)

## Mathematical Background

A summary of the above paper is presented as follows. The logistic regression model is one with continuous input $X \in \mathbb{R}$ and outputs a binary classifier $y \in \{0, 1\}$ (sometimes defined as {-1, +1} instead):

$$y = \begin{cases}
    1 & \text{ with prob. } \dfrac{1}{1+\exp(-\beta\langle X, \theta \rangle)} \\[13pt]
    0 & \text{ with prob. } \dfrac{1}{1+\exp(\beta\langle X, \theta \rangle)}
\end{cases}$$

where $X = \{x_1, x_2, x_3, ..., x_n\}$, where $x_i \in \mathbb{R}^d$ and $\theta \in \mathbb{R}^d, ||\theta|| = 1.$

The value $\beta$, considered the inverse temperature, is what governs the \textit{signal-to-noise ratio}; when $\beta = 0$, we have pure noise; when $\beta = \infty$, we simply have a linear classifier which denotes where $x$ lies on the hyperplane. Then, in this project, we seek to validate the paper above's claim that the sample complexity $n^*(d, \beta, \epsilon)$, which is the smallest sample size $n$ such that $||\theta - \hat{\theta}|| < \epsilon$, satisfies

$$n^*(d, \beta, \epsilon) \asymp \begin{cases}
    \dfrac{d}{\beta^2\epsilon^2} & \text{ if } \beta ≲ 1 \text{ (high temperatures);} \\[13pt]
    \dfrac{d}{\beta\epsilon^2} & \text{ if } 1 ≲ \beta ≲ \dfrac{1}{\epsilon} \text{ (medium temperatures);} \\[13pt]
    \dfrac{d}{\beta^2\epsilon^2} & \text{ if } \beta ≳ \dfrac{1}{\epsilon} \text{ (low temperatures);}
\end{cases}$$

## Data Generation

To generate the dataset, I generated $\theta \sim N(0, 1)$ and $X \sim U[-1, 1]$, producing an output binary classification label $y = \text{Bern}(\sigma(X^T \cdot \theta))$ where $\sigma(\eta)$ is the sigmoid function $(1+\exp(-\eta))^{-1}$.
```# Dataset generation
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
```

## Basic Scikit-Learn Model

First, I started out with a basic sci-kit model estimating a randomly generated $\theta$ when varying $n$ (sample size) and $||\theta||$ (norm of the parameter vector). I used the `LogisticRegression` class from `sklearn.linear_model` to produce various graphs with the RMSE. The sample code can be found in the folder. This model proved to not be enough as the plain setup yielded results $\hat{\theta}$ which did not approach the true norm of the parameter vector. Instead, we chose to move to [CVXPY](https://www.cvxpy.org/index.html) (convex optimization programming library).

## CVXPY

CVXPY is a library which simplifies convex optimization problems (in our case, minimizing negative logistic loss). The setup for the problem is relatively simple:
```# Initialize the CVXPY maximization problem.
def setup(self):
    self.weights = cp.Variable(NUM_FEATURES)
    log_likelihood = cp.sum(
        cp.multiply(np.ndarray.flatten(self.y), self.X @ self.weights) - cp.logistic(self.X @ self.weights)
    )
    constraints = [cp.norm(self.weights) <= self.beta]
    return cp.Problem(cp.Maximize(log_likelihood/self.n), constraints)
```

## Results

TBD