import numpy as np

# Activation function (Sigmoid) and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Sample dataset (2 inputs, 1 output)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input (4 samples, 2 features each)
y = np.array([[0], [1], [1], [0]])  # XOR problem (Expected output)

# Initialize weights and biases randomly
np.random.seed(42)
input_size = 2
hidden_size = 2  # One hidden layer with 2 neurons
output_size = 1

W1 = np.random.randn(input_size, hidden_size)
B1 = np.random.randn(1, hidden_size)
W2 = np.random.randn(hidden_size, output_size)
B2 = np.random.randn(1, output_size)

# Hyperparameters
learning_rate = 0.1
num_epochs = 10000

# Training with backpropagation
for epoch in range(num_epochs):
    # Forward Pass
    hidden_input = np.dot(X, W1) + B1
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, W2) + B2
    final_output = sigmoid(final_input)
    
    # Compute error
    error = y - final_output
    
    # Backpropagation
    d_output = error * sigmoid_derivative(final_output)  # Derivative of loss w.r.t output layer
    d_hidden = d_output.dot(W2.T) * sigmoid_derivative(hidden_output)  # Chain rule for hidden layer
    
    # Gradient Descent - Update Weights and Biases
    W2 += hidden_output.T.dot(d_output) * learning_rate
    B2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    W1 += X.T.dot(d_hidden) * learning_rate
    B1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate
    
    # Print loss every 1000 epochs
    if epoch % 1000 == 0:
        loss = np.mean(np.abs(error))
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# Final Output
print("Final Predictions:")
print(final_output)



# Move data to device (CPU/GPU)
#X_batch = X_batch.to(device, non_blocking=True) 
#y_batch = y_batch.to(device, non_blocking=True)

# Step 1: Zero accumulated gradients (prevents mixing old gradients)
#optimizer_obj.zero_grad()

# Step 2: Forward Pass - Compute Predictions
#y_pred = model(X_batch)  # \( \hat{y} = f(W, X) \)

# Step 3: Compute Loss
#loss_res = loss_obj(y_pred, y_batch)  # \( L = \text{loss}(\hat{y}, y) \)

# Step 4: Backpropagation - Compute Gradients
#loss_res.backward()  # \( \nabla L = \frac{\partial L}{\partial W} \)

# Step 5: Update Weights Using Gradient Descent
#optimizer_obj.step()  # \( W = W - \alpha \nabla L \)

# Step 6: Accumulate Loss
#total_loss += loss_res.item()