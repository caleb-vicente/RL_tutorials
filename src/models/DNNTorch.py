import torch
import torch.nn as nn


class DNN(nn.Module):
    def __init__(self, input_dim, output_dim, layers_sizes, activation_functions):
        super(DNN, self).__init__()

        self.layers_sizes = [input_dim] + layers_sizes + [output_dim]

        if not len(activation_functions) == (len(self.layers_sizes) - 1):
            raise ValueError("Number of activation functions must match number of layers")

        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        for i in range(len(self.layers_sizes) - 1):
            self.layers.append(nn.Linear(self.layers_sizes[i], self.layers_sizes[i + 1]))

            if activation_functions[i] == 'relu':
                self.activations.append(nn.ReLU())
            elif activation_functions[i] == 'leaky_relu':
                self.activations.append(nn.LeakyReLU())
            elif activation_functions[i] == 'tanh':
                self.activations.append(nn.Tanh())
            elif activation_functions[i] == 'softmax':
                self.activations.append(nn.Softmax())
            else:
                raise ValueError("Invalid activation function")

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.activations[i](self.layers[i](x))

        x = self.layers[-1](x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_outputs)
        self.lrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        return self.softmax(self.fc3(x))
