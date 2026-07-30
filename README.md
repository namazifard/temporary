# Backpropagation example

Here is a simple Python implementation of a backpropagation step for a 2-layer neural network using NumPy:

```python
import numpy as np


def backpropagation(X, y, params, learning_rate=0.01):
    """Perform one forward and backward pass for a 2-layer neural network."""
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]

    # Forward pass
    z1 = X @ W1 + b1
    a1 = np.maximum(z1, 0)  # ReLU
    z2 = a1 @ W2 + b2
    a2 = 1 / (1 + np.exp(-z2))  # sigmoid

    # Loss (binary cross-entropy)
    loss = -(y * np.log(a2 + 1e-8) + (1 - y) * np.log(1 - a2 + 1e-8)).mean()

    # Backward pass
    da2 = (a2 - y) / y.size
    dW2 = a1.T @ da2
    db2 = da2.sum(axis=0)

    da1 = da2 @ W2.T
    dz1 = da1 * (z1 > 0)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)

    # Gradient descent update
    grads = {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2,
    }

    params["W1"] -= learning_rate * dW1
    params["b1"] -= learning_rate * db1
    params["W2"] -= learning_rate * dW2
    params["b2"] -= learning_rate * db2

    return loss, grads, params
```

Example:

```python
X = np.array([[0.2, 0.4], [0.8, 0.1]])
y = np.array([[1], [0]])

params = {
    "W1": np.random.randn(2, 3),
    "b1": np.zeros((1, 3)),
    "W2": np.random.randn(3, 1),
    "b2": np.zeros((1, 1)),
}

loss, grads, params = backpropagation(X, y, params, learning_rate=0.01)
print(loss)
```
