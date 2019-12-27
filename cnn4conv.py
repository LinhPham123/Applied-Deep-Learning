import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self, channels: int, class_count: int, dropout: float):
        super(CNN, self).__init__()

        self.class_count = class_count
        self.drop_out = nn.Dropout(dropout)
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=(43,21)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer1.apply(self.init_weights_bias)

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(2,2), padding=(43,21)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), padding=(1,1)),
            # self.drop_out
        )
        self.layer2.apply(self.init_weights_bias)
       
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(2,2), padding=(22, 11)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer3.apply(self.init_weights_bias)

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # self.drop_out
        )
        self.layer4.apply(self.init_weights_bias)

        self.layer5 = nn.Sequential(
            nn.Linear(15488, 1024),
            nn.Sigmoid(),
            # self.drop_out
        )
        self.layer5.apply(self.init_weights_bias)
        
        self.out = nn.Linear(1024, 10) #don't need softMax here because crossEntropy already have softMax
        self.initialise_layer(self.out)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        
        x = self.layer1(images)
        
        x = self.layer2(x)
        x = self.drop_out(x)
        
        x = self.layer3(x)
       
        x = self.layer4(x)
        x = self.drop_out(x)
        
        x = torch.flatten(x, start_dim=1)
       
        x = self.layer5(x)
        x = self.drop_out(x)

        x = self.out(x)
        return x

    def init_weights_bias(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            self.initialise_layer(m)
   
    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)
