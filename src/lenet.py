import torch as t
import numpy as np
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        # MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        # Conv2d(1, 6, 5)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Conv2d(6, 16, 5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Linear(16*5*5, 120)
        self.fc1 = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU()
        )
        # Linear(120, 84)
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        # Linear(84, 10)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Conv2d(1, 6, 5)
        x = self.conv1(x)
        # Conv2d(6, 16, 5)
        x = self.conv2(x)
        # Linear(16*5*5, 120)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        # Linear(120, 84)
        x = self.fc2(x)
        # Linear(84, 10)
        x = self.fc3(x)
        return x