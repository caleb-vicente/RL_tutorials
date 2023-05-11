import torch
import torch.nn as nn

class DNN(nn.Module):
    # TODO: dnn should let the user to configure number layers, activations, etc
    def __init__(self, input_dim, output_dim):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x