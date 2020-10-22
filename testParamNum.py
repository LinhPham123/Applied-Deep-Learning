import torch
import numpy
from torch import nn
from cnnNoStride import *

height = int(input("Enter height: "))
width = int(input("Enter width: "))
channel = int(input("Enter number of channels: "))

model = CNNNoStride(height, width, channel)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

total_param = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        num_param = numpy.prod(param.size())
        if param.dim() > 1:
            print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
        else:
            print(name, ':', num_param)
        total_param += num_param
print(total_param)