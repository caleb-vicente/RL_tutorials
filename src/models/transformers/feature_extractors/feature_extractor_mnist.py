import torch.nn as nn


class FeatureExtractorConvolutional(nn.Module):
    def __init__(self, ch_in, conv1_ch, conv2_ch, conv3_ch, conv4_ch):
        super(FeatureExtractorConvolutional, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, conv1_ch, kernel_size=(4, 4))
        self.conv2 = nn.Conv2d(conv1_ch, conv2_ch, kernel_size=(4, 4))
        self.conv3 = nn.Conv2d(conv2_ch, conv3_ch, kernel_size=(4, 4))
        self.conv4 = nn.Conv2d(conv3_ch, conv4_ch, kernel_size=(4, 4))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = x.squeeze()
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        return x
