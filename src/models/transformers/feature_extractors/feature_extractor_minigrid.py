import torch.nn as nn


class FeatureExtractorMiniGrid(nn.Module):
    def __init__(self, ch_in, conv1_ch, conv2_ch):
        super(FeatureExtractorMiniGrid, self).__init__()
        self.ch_in = ch_in
        self.conv1_ch = conv1_ch
        self.conv2_ch = conv2_ch
        self.conv1 = nn.Conv2d(self.ch_in, self.conv1_ch, kernel_size=(1, 1), padding=0)  # A
        self.conv2 = nn.Conv2d(self.conv1_ch, self.conv2_ch, kernel_size=(1, 1), padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x
