from torch import nn

class CNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int, class_count: int, dropout: float):
        super(CNN, self).__init__()
        self.input_shape = ImageShape(height=height, width=width, channels=channels)
        self.class_count = class_count

        self.drop_out = nn.Dropout(dropout)

        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), stride=(2,2)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
       
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(2,2)),
            nn.BatchNorm2d(64),
            nn.ReLU,
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.5)
        )
       
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(2,2)),
            nn.BatchNorm2d(64),
            nn.ReLU
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(2,2)),
            nn.BatchNorm2d(64),
            nn.ReLU,
            nn.Dropout(0.5)
        )

        self.layer4 = nn.Sequential(
            nn.Linear(15488, 1024),
            nn.Sigmoid(),
            nn.Dropout(0.5)
        )
        
        self.out = nn.Sequential(
            nn.Linear(1024, 10),
            nn.Softmax
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.conv1(images)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        ## TASK 2-2: Pass x through the second convolutional layer
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        ## TASK 3-2: Pass x through the second pooling layer
        x = self.pool2(x)
        ## TASK 4: Flatten the output of the pooling layer so it is of shape
        ##         (batch_size, 4096)
        x = torch.flatten(x, start_dim=1)
        x = self.drop_out(x)
        ## TASK 5-2: Pass x through the first fully connected layer
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.drop_out(x)
        ## TASK 6-2: Pass x through the last fully connected layer
        x = self.fc2(x)
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)
