# Predict whether a student passes or fails an exam based on:
# Hours studied, Hours slept, Classes attended
import numpy as np
import matplotlib.pyplot as plt

# Training data
# Columns: study hours, sleep hours, classes attended
X = np.array([
    [1, 6, 3],
    [2, 7, 4],
    [3, 5, 5],
    [4, 8, 6],
    [5, 6, 7],
    [6, 7, 8],
    [2, 8, 3],
    [5, 5, 8],
    [3, 7, 4],
    [6, 8, 9]
], dtype=float)

# Labels: 0 = Fail, 1 = Pass
y = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])

# Step 1: Standardize features
mean = X.mean(axis=0)
std = X.std(axis=0)
X_scaled = (X - mean) / std

# Step 2: Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Step 3: Initialize parameters
n, features = X_scaled.shape

w = np.zeros(features)
b = 0.0

learning_rate = 0.1
epochs = 5000
loss_history = []

# Step 4: Gradient descent
for epoch in range(epochs):

    # Forward pass
    z = X_scaled @ w + b
    predictions = sigmoid(z)

    # Binary cross-entropy loss
    loss = -np.mean(
        y * np.log(predictions + 1e-12) +
        (1 - y) * np.log(1 - predictions + 1e-12)
    )
    loss_history.append(loss)

    # Calculate gradients
    error = predictions - y

    dw = X_scaled.T @ error / n
    db = np.mean(error)

    # Update parameters
    w -= learning_rate * dw
    b -= learning_rate * db

# Step 5: Evaluate training predictions
probabilities = sigmoid(X_scaled @ w + b)
y_pred = (probabilities >= 0.5).astype(int)

accuracy = np.mean(y_pred == y)

print("Weights:", np.round(w, 3))
print("Bias:", round(b, 3))
print(f"Training accuracy: {accuracy:.2%}")

# Step 6: Predict a new student
new_student = np.array([[4, 7, 6]], dtype=float)
new_scaled = (new_student - mean) / std

probability = sigmoid(new_scaled @ w + b)[0]

print("Probability of passing:", round(probability, 3))
print("Prediction:", "Pass" if probability >= 0.5 else "Fail")

# Step 7: Plot loss history
plt.figure(figsize=(7, 4))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("Logistic Regression Training Loss")
plt.grid(alpha=0.3)
plt.show()
