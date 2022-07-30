# TODO: define the convolutional neural network architecture

from cmath import tan
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # Calculate feature map sizes by:
        # new width after first convolution:
        # = (W (224) - kernel_size + 2 Padding) / Stride   +  1
        # and after pooling:
        # new width = (W (220) - kernel_size) / stride  +  1

        # from 224 x 224 x 1Channel
        # to   110 x 100 x 1Channel x 32 Feature maps
        # and  54 x 54 x 1Channel x 32 Feature maps

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            )

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )

        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.AvgPool2d(2,2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
      
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*12*12, 1024),
            nn.Tanh(),
            nn.Dropout1d(0.01),
            nn.Linear(1024, 136)
        )

        self.simple = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(10816,4096),
            nn.ReLU(),
            nn.Linear(4096,1024),
            nn.Linear(1024, 136)
        )
        torch.nn.init.xavier_uniform_(self.conv_layer1[0].weight)
        torch.nn.init.xavier_uniform_(self.conv_layer2[0].weight)
        torch.nn.init.xavier_uniform_(self.conv_layer3[0].weight)
        torch.nn.init.xavier_uniform_(self.conv_layer4[0].weight)
        torch.nn.init.xavier_normal_(self.regressor[1].weight)
        torch.nn.init.xavier_normal_(self.regressor[4].weight)
       # torch.nn.init.xavier_normal_(self.regressor[5].weight)

        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

    def forward(self, x):
        #x = self.conv_layer1(x)
       # x = self.conv_layer2(x)
       # x = self.conv_layer3(x)
      #  x = self.conv_layer4(x)
      #  x = self.regressor(x)
        return self.simple(x)
