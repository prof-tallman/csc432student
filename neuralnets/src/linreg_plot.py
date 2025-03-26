import numpy as np
import matplotlib.pyplot as plt

# Sample dataset (x: input, y: actual output)
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])  # Linear relationship y = 2x

# Initialize parameters
w = np.random.randn()  # Random weight
b = np.random.randn()  # Random bias

# Hyperparameters
learning_rate = 0.01
num_epochs = 1000

# Number of data points
n = len(x)

# Store loss values for visualization
loss_history = []

# Gradient Descent
for epoch in range(num_epochs):
    # Predictions
    y_pred = w * x + b
    
    # Compute error
    error = y - y_pred
    
    # Compute gradients
    dw = (-2/n) * np.sum(error * x)
    db = (-2/n) * np.sum(error)
    
    # Update weights and bias
    w -= learning_rate * dw
    b -= learning_rate * db
    
    # Compute and store loss
    mse = np.mean(error**2)
    loss_history.append(mse)
    
    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: MSE = {mse:.4f}, w = {w:.4f}, b = {b:.4f}")

# Final parameters
print(f"Final model: y = {w:.4f}x + {b:.4f}")

# Plot loss over epochs
plt.plot(range(num_epochs), loss_history, label='MSE Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Over Time')
plt.legend()
plt.show()