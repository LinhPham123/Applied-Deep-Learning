import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

def calculate_padding(input_size: int, output_size: int, filter_size: int, stride_size: int):
    return int(np.ceil(((output_size - 1) * stride_size - input_size + filter_size) / 2))

class CNN(nn.Module):
    def __init__(self, input_height: int, input_width: int, input_channels: int):
        super(CNN, self).__init__()
        
        self.drop_out = nn.Dropout2d(0.5)

        pad_h = calculate_padding(input_height, input_height, filter_size=3, stride_size=2)
        pad_w = calculate_padding(input_width, input_width, filter_size=3, stride_size=2)
        
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=(pad_h, pad_w), bias=False) #43, 21
        self.initialise_layer(self.conv1)
        self.bn32_1 = nn.BatchNorm2d(32)
          
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=(pad_h, pad_w), bias=False) #43, 21
        self.initialise_layer(self.conv2)
        self.bn32_2 = nn.BatchNorm2d(32)
        
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2, 2), padding=(1,1)) #ceil_mode=True

        pad_h = int(np.ceil(pad_h / 2))
        pad_w = int(np.ceil(pad_w / 2))
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(2,2), padding=(pad_h, pad_w), bias=False) #22, 11
        self.initialise_layer(self.conv3)
        self.bn64_1 = nn.BatchNorm2d(64)
       
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(2,2), padding=(1,1), bias=False) #1, 1
        self.initialise_layer(self.conv4)
        self.bn64_2 = nn.BatchNorm2d(64)
      
        new_h = int(np.ceil(input_height / 4))
        new_w = int(np.ceil(input_width / 4))
    
        self.fc = nn.Linear(new_h * new_w * 64, 1024)
        self.initialise_layer(self.fc)
   
        self.out = nn.Linear(1024, 10)
        self.initialise_layer(self.out)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.conv1(images)
        x = self.bn32_1(x)
        x = F.relu(x)

        x = self.drop_out(x)
        x = self.conv2(x)
        x = self.bn32_2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.bn64_1(x)
        x = F.relu(x)

        x = self.drop_out(x)
        x = self.conv4(x)
        x = self.bn64_2(x)
        x = F.relu(x)
        
        x = torch.flatten(x, start_dim=1)
        x = self.drop_out(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        
        x = self.out(x)
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias") and layer.bias is not None:
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')


# shape = ImageShape(height=145,width=41,channels=1)
# from torchsummary import summary
# model = CNN(shape, 10)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
# summary(model,(1,145,41), 32)
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

