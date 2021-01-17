import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self ,in_dim ,n_channels ,length):
        super(ConvBlock,self).__init__()

    def forward(self,x):
        pass


class Bottleneck(nn.Module) :

    def __init__(self,in_planes,planes,stride):
        super(Bottleneck,self).__init__()
        self.conv1 = nn.Conv2d(in_planes,planes,kernel_size=3,stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(in_planes,planes,kernel_size=3,stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(in_planes,planes,kernel_size=3,stride=stride)



    def forward(self,x):
        pass

class ResNet(nn.Module):
    
    def __init__(self,length,blocks,n_classes=10):
        super(ResNet,self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = self._make_layer(ConvBlock,)

    def forward(self,x):
        pass

    def _make_layer(self,block,in_dim,n_channels,length):
        pass




def resnet18():
    return ResNet(18,[2,2,2,2])

