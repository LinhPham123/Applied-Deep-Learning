import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class CNNNoStride(nn.Module):
    def __init__(self, input_height: int, input_width: int, input_channels: int):
        super(CNNNoStride, self).__init__()

        self.dropout2d = nn.Dropout2d(p=0.5)

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=(3, 3), padding=(1, 1), bias = False)
        self.initialise_layer(self.conv1)
        self.bn_conv1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=(1, 1), bias = False)
        self.initialise_layer(self.conv2)
        self.bn_conv2 = nn.BatchNorm2d(32)

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(1,1))

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1), bias = False)
        self.initialise_layer(self.conv3)
        self.bn_conv3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1), bias = False)
        self.initialise_layer(self.conv4)
        self.bc_conv4 = nn.BatchNorm2d(64)

        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(1,1))

        new_h = int(np.ceil(input_height / 4))
        new_w = int(np.ceil(input_width / 4))
       
        self.fc1 = nn.Linear(new_h * new_w * 64, 1024)
        self.initialise_layer(self.fc1)
    
        self.fc2 = nn.Linear(1024, 10)
        self.initialise_layer(self.fc2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn_conv1(self.conv1(images)))
        x = self.dropout2d(x)
        x = F.relu(self.bn_conv2(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.bn_conv3(self.conv3(x)))
        x = self.dropout2d(x)
        x = F.relu(self.bc_conv4(self.conv4(x)))
        x = self.pool2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout2d(x)
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

      
    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias") and type(layer.bias) != type(None):   
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
