import torch
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DNN, self).__init__()

        #self.normalization = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 100)
        self.fc3 = nn.Linear(100, output_dim)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        #x = self.normalization(x)
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x
