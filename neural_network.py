import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# ==========================================
# 1. Dataset Generation
# ==========================================
print("📊 Generating dataset...")
# Creating a non-linear dataset (two interleaving half circles)
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
y = y.reshape(-1, 1) # Reshape y to be a column vector

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 2. Activation Functions & Derivatives
# ==========================================
# Sigmoid: Maps values between 0 and 1 (used for output layer)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# ReLU: Returns x if x > 0, else 0 (used for hidden layers)
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# ==========================================
# 3. Neural Network Class (Mini-TensorFlow)
# ==========================================
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.lr = learning_rate
        
        # Initialize weights and biases randomly
        # W1: Weights between Input and Hidden Layer
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        
        # W2: Weights between Hidden and Output Layer
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
        self.loss_history = []

    def forward(self, X):
        # Layer 1 (Hidden)
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = relu(self.Z1)
        
        # Layer 2 (Output)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)
        
        return self.A2

    def backward(self, X, y):
        m = X.shape[0] # Number of samples
        
        # Output Layer Error
        dZ2 = self.A2 - y
        dW2 = (1 / m) * np.dot(self.A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Hidden Layer Error (Backpropagation)
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * relu_derivative(self.A1)
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Update Weights and Biases (Gradient Descent)
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, y, epochs=2000):
        print("🧠 Training started...")
        for epoch in range(epochs):
            # 1. Forward Pass
            predictions = self.forward(X)
            
            # 2. Compute Loss (Binary Cross Entropy)
            loss = -np.mean(y * np.log(predictions + 1e-8) + (1 - y) * np.log(1 - predictions + 1e-8))
            self.loss_history.append(loss)
            
            # 3. Backward Pass
            self.backward(X, y)
            
            # Print progress
            if epoch % 200 == 0:
                print(f"Epoch {epoch} | Loss: {loss:.4f}")
        print("✅ Training complete!")

    def predict(self, X):
        predictions = self.forward(X)
        return (predictions > 0.5).astype(int)

# ==========================================
# 4. Train and Evaluate
# ==========================================
# Initialize Network: 2 Inputs -> 10 Hidden Neurons -> 1 Output
nn = NeuralNetwork(input_size=2, hidden_size=10, output_size=1, learning_rate=0.1)

# Train the model
nn.train(X_train, y_train, epochs=2000)

# Calculate Accuracy
test_predictions = nn.predict(X_test)
accuracy = np.mean(test_predictions == y_test) * 100
print(f"🎯 Test Accuracy: {accuracy:.2f}%")

# ==========================================
# 5. Plot Loss Curve
# ==========================================
plt.plot(nn.loss_history, color='blue')
plt.title("Training Loss over Epochs (Learning Curve)")
plt.xlabel("Epochs")
plt.ylabel("Loss (Binary Cross Entropy)")
plt.grid()
plt.show()