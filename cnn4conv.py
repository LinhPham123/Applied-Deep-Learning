import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self, channels: int, class_count: int, dropout: float):
        super(CNN, self).__init__()

        self.class_count = class_count
        self.drop_out = nn.Dropout(dropout)

        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), stride=(2,2)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.initialise_layer(self.layer1)
       
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(2,2)),
            nn.BatchNorm2d(64),
            nn.ReLU,
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.5)
        )
        self.initialise_layer(self.layer2)
       
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(2,2)),
            nn.BatchNorm2d(64),
            nn.ReLU
        )
        self.initialise_layer(self.layer3)

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(2,2)),
            nn.BatchNorm2d(64),
            nn.ReLU,
            nn.Dropout(0.5)
        )
        self.initialise_layer(self.layer4)

        self.layer5 = nn.Sequential(
            nn.Linear(15488, 1024),
            nn.Sigmoid(),
            nn.Dropout(0.5)
        )
        self.initialise_layer(self.layer5)
        
        self.out = nn.Linear(1024, 10)
        self.initialise_layer(self.out)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.layer1(images)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(x, start_dim=1)
        x = self.layer5(x)
        x = self.out(x)
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)
