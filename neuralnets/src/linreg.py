import numpy as np
import time

# Sample dataset that is perfectly linear (i.e., unrealistic)
# Linear relationship y = 2x + 1
x = np.array([0, 1, 2, 3, 4])
y = np.array([(2*xi + 1) for xi in x])
print(f"x: {x}")
print(f"y: {y}")

# Initialize parameters
w = np.random.randn()  # Random weight
b = np.random.randn()  # Random bias
print(f"Initial guesses are w={w:.6f} and b={b:.6f}")
input("Press <ENTER> to begin")

# Hyperparameters
learning_rate = 0.01
num_epochs = 1000

# Number of samples
n = len(x)

# Gradient Descent
debug_sleep_delay = 10
for epoch in range(num_epochs):

    # Calculate gradients for this step
    y_pred = w * x + b
    error = y - y_pred
    dw = (-2/n) * np.sum(error * x)
    db = (-2/n) * np.sum(error)
    
    # Update weights and bias
    w -= learning_rate * dw
    b -= learning_rate * db
    
    # Print examples
    print(f"Epoch {epoch:>3}: w = {w:.4f} (Δw={learning_rate:.2f}*{dw:.4f}) b = {b:.4f} (Δb={learning_rate:.2f}*{db:.4f})")
    time.sleep(debug_sleep_delay)
    debug_sleep_delay *= 0.5

# Final parameters
print(f"Final model: y = {w:.4f}x + {b:.4f}")


