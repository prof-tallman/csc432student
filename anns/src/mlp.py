import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Fake the data, which has already normalized
# Credit Score, Debt to Income Ratio, Employment Stability
# 1 = Loan Approved, 0 = Loan Denied
X = np.array([
    [0.8, 0.2, 1.0],  # Likely approved 750 credit score
    [0.5, 0.5, 0.5],  # Uncertain       550 CS
    [0.9, 0.1, 0.8],  # Likely approved 820 CS
    [0.4, 0.7, 0.2]   # Likely denied   400 CS
])
y = np.array([1, 0, 1, 0])

# Standardize the input data
# Important for MLP even if the data is already normalized
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Create the Neural Network with three different solvers
mlp1 = MLPClassifier(
    hidden_layer_sizes=(4, 3),  # two hidden layers: 4 neurons → 3 neurons
    activation='relu',          # hidden layers only; output auto to sigmoid
    solver='adam',              # 'adam' 'sgd' and 'lbfgs'
    max_iter=1000,              # increase iterations to ensure convergence
    random_state=42,            # repeatability
    learning_rate_init=0.01     # slower than default to help convergence
)

mlp2 = MLPClassifier(
    hidden_layer_sizes=(4, 3),  # two hidden layers: 4 neurons → 3 neurons
    activation='relu',          # hidden layers only; output auto to sigmoid
    solver='sgd',               # 'adam' 'sgd' and 'lbfgs'
    max_iter=500,               # increase iterations to ensure convergence
    random_state=42,            # repeatability
    learning_rate_init=0.01     # slower than default to help convergence
)

mlp3 = MLPClassifier(
    hidden_layer_sizes=(4, 3),  # two hidden layers: 4 neurons → 3 neurons
    activation='relu',          # hidden layers only; output auto to sigmoid
    solver='lbfgs',             # 'adam' 'sgd' and 'lbfgs'
    max_iter=500,               # increase iterations to ensure convergence
    random_state=42,            # repeatability
    learning_rate_init=0.01     # slower than default to help convergence
)

# Train the neural networks
mlp1.fit(X, y)
mlp2.fit(X, y)
mlp3.fit(X, y)

# New loan applicant already normalized but needs to be standardized
X_new = np.array([[0.5, 0.1, 0.8]])
X_new = scaler.transform(X_new)

# Predict a new loan applicant
y_prob1 = mlp1.predict_proba(X_new)
y_pred1 = mlp1.predict(X_new)
print(f"Adam Prediction Probabilities: {y_prob1}")
print(f"Adam Final Decision: {'Approved' if y_pred1[0] == 1 else 'Denied'}")
y_prob2 = mlp2.predict_proba(X_new)
y_pred2 = mlp2.predict(X_new)
print(f"SGD Prediction Probabilities: {y_prob2}")
print(f"SGD Final Decision: {'Approved' if y_pred2[0] == 1 else 'Denied'}")
y_prob3 = mlp3.predict_proba(X_new)
y_pred3 = mlp3.predict(X_new)
print(f"LBFGS Prediction Probabilities: {y_prob3}")
print(f"LBFGS Final Decision: {'Approved' if y_pred3[0] == 1 else 'Denied'}")
