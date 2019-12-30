import torch
from torch import nn
from torch.nn import functional as F

class CNN(nn.Module):
    def __init__(self, channels: int, class_count: int):
        super(CNN, self).__init__()

        self.class_count = class_count
        self.drop_out = nn.Dropout(0.5)
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=(43,21))
        self.initialise_layer(self.conv1)
        self.bn32_1 = nn.BatchNorm2d(32)
        
         
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=(43,21))
        self.initialise_layer(self.conv2)
        self.bn32_2 = nn.BatchNorm2d(32)
 
        self.pool = nn.MaxPool2d(kernel_size=(2,2), padding=(1,1))

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(2,2), padding=(22, 11))
        self.initialise_layer(self.conv3)
        self.bn64_1 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.initialise_layer(self.conv4)
        self.bn64_2 = nn.BatchNorm2d(64)

        self.fc = nn.Linear(15488, 1024)
        self.initialise_layer(self.fc)
   
        self.out = nn.Linear(1024, 10) #don't need softMax here because crossEntropy already have softMax
        self.initialise_layer(self.out)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.conv1(images)
        x = self.bn32_1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn32_2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.drop_out(x)
        
        x = self.conv3(x)
        x = self.bn64_1(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn64_2(x)
        x = F.relu(x)
        x = self.drop_out(x)
        
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        x = self.drop_out(x)

        x = self.out(x)
    
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)


# model = CNN(1, 10)


# import numpy
# total_param = 0
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         num_param = numpy.prod(param.size())
#         if param.dim() > 1:
#             print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
#         else:
#             print(name, ':', num_param)
#         total_param += num_param
# print(total_param)