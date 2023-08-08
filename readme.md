# RSRI Projects`

## Logistic Regression

For binary classification tasks, one family of functions is the logistic functions, defined by

$$\phi_{\text{sig}}(z) = \dfrac{1}{1+e^{-z}}$$

and a prediction is formed with $z$ as

$$z = \left(\sum_{i=1}^n w_ix_i\right) + b$$

with $w$ as the weights, $x$ as the inputs and $b$ as the bias. The prediction then becomes

$$\hat{y} = \phi(z)$$

We will then use a basic gradient descent algorithm to optimize our weights $w$:

$$w_{t+1} = \theta_t - \eta\nabla L(h(x;w), y)$$ 

with $L$ as our loss function. A possible loss function can just be the square loss

$$L(h(x;w), y) = (h(x;w) - y)^2$$