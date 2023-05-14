import torch
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, input_dim, output_dim, layers_sizes, activation_function):
        super(DNN, self).__init__()

        self.layers_sizes = [input_dim] + layers_sizes + [output_dim]
        self.activation_function = activation_function

        self.layers = nn.ModuleList()

        for i in range(len(self.layers_sizes) - 1):
            self.layers.append(nn.Linear(self.layers_sizes[i], self.layers_sizes[i+1]))

        if self.activation_function == 'relu':
            self.activation = nn.ReLU()
        elif self.activation_function == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif self.activation_function == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError("Invalid activation function")

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))

        x = self.layers[-1](x)
        return x

