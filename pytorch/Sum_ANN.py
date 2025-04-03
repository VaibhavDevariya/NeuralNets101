import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Neural net for finding the summation of 2 numbers
class SumNN(nn.Module):
    def __init__(self):
        super(SumNN, self).__init__()
        self.fc1 = nn.Linear(2, 16)  # Hidden layer with 16 neurons
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)  # Output layer
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create model
model = SumNN()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Generate training data
np.random.seed(42)
x_train = np.random.rand(1000, 2) * 10  # Random numbers between 0 and 10
y_train = np.sum(x_train, axis=1, keepdims=True)  # Compute sum

# Convert to tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# Train the model
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'Epoch [{epoch}/1000], Loss: {loss.item():.4f}')

# Test the model
x_test = torch.tensor([[4.5, 3.2]], dtype=torch.float32)
predicted_sum = model(x_test).item()
print(f'Predicted sum: {predicted_sum}, Actual sum: {sum(x_test.numpy()[0])}')
