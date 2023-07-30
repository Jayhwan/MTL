import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def generate_data(num_samples):
    X = np.random.randint(1, 100, size=(num_samples, 2))
    y_multiply = X[:, 0] * X[:, 1]
    y_divide = X[:, 0] / X[:, 1]
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y_multiply, dtype=torch.float32), torch.tensor(y_divide, dtype=torch.float32)

# Hard Parameter Sharing 모델 정의
class HardParameterSharingModel(nn.Module):
    def __init__(self, shared_size=16, task_size=8):
        super(SoftParameterSharingModel, self).__init__()

        # 공유 파라미터
        self.shared_layer = nn.Linear(2, shared_size)

        # Task-specific 파라미터
        self.multiply_output = nn.Linear(shared_size, task_size)
        self.divide_output = nn.Linear(shared_size, task_size)

    def forward(self, x):
        x_shared = torch.relu(self.shared_layer(x))
        multiply_result = self.multiply_output(x_shared)
        divide_result = self.divide_output(x_shared)
        return multiply_result, divide_result

# Soft Parameter Sharing 모델
class SoftParameterSharingModel(nn.Module):
    def __init__(self, shared_size=16, task_size=8, l2_reg=0.01):
        super(SoftParameterSharingModel, self).__init__()

        # 공유 파라미터
        self.shared_layer = nn.Linear(2, shared_size)

        # Task-specific 파라미터
        self.multiply_output = nn.Linear(shared_size, task_size)
        self.divide_output = nn.Linear(shared_size, task_size)

        self.l2_reg = l2_reg

    def forward(self, x):
        x_shared = torch.relu(self.shared_layer(x))
        multiply_result = self.multiply_output(x_shared)
        divide_result = self.divide_output(x_shared)
        return multiply_result, divide_result

    def compute_l2_loss(self):
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)
        return self.l2_reg * l2_loss



X_train, y_multiply_train, y_divide_train = generate_data(1000)
X_test, y_multiply_test, y_divide_test = generate_data(200)

model = SoftParameterSharingModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습
num_epochs = 20
batch_size = 32
num_batches = len(X_train) // batch_size

for epoch in range(num_epochs):
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        inputs = X_train[start_idx:end_idx]
        multiply_targets = y_multiply_train[start_idx:end_idx]
        divide_targets = y_divide_train[start_idx:end_idx]

        optimizer.zero_grad()

        multiply_outputs, divide_outputs = model(inputs)
        multiply_loss = criterion(multiply_outputs.view(-1), multiply_targets)
        divide_loss = criterion(divide_outputs.view(-1), divide_targets)

        l2_loss = model.compute_l2_loss()

        loss = multiply_loss + divide_loss + l2_loss
        loss.backward()

        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")