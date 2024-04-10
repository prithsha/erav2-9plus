from typing import Any, Callable, List, Optional, Type
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def conv3x3(in_channels: int, out_channels: int, stride: int = 1, 
             padding = 0, dilation: int = 1, groups: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels,out_channels,kernel_size=3,
                    stride=stride,padding=padding,
                    dilation=dilation, groups=groups, bias=False )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

def get_3x3_basic_conv_2d_layer(in_channels, out_channels, stride=1, padding = 0, dilation = 1, drop_out=0.0):
    conv_layer = nn.Sequential(
        conv3x3(in_channels,out_channels, stride=stride,padding=padding,dilation=dilation),        
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),        
        nn.Dropout(drop_out)
    )
    return conv_layer

def get_1x1_basic_conv_2d_layer(in_channels, out_channels, stride=1):
    conv_layer = nn.Sequential(
        conv1x1(in_channels,out_channels, stride=stride),        
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

    return conv_layer

def get_custom_conv_with_max_pool_layer(in_channels, out_channels, stride=1, padding = 1):
    max_pool_layer = nn.Sequential(
        conv3x3(in_channels,out_channels, stride=stride,padding=padding),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),    
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
    return max_pool_layer



class BasicResBlock(nn.Module):

    def __init__( self, in_channels: int, out_channels: int, stride: int = 1, padding: int = 1 ) -> None:
        super().__init__()

        self.conv_with_max_pool = get_custom_conv_with_max_pool_layer(in_channels,out_channels,stride=1, padding=1)
        self.conv1 = get_3x3_basic_conv_2d_layer(in_channels=in_channels, out_channels=out_channels, stride=stride, padding=padding)
        self.conv2 = get_3x3_basic_conv_2d_layer(in_channels=out_channels, out_channels=out_channels, stride=stride, padding=padding)

        self.size_norm = get_1x1_basic_conv_2d_layer(in_channels=out_channels, out_channels=out_channels, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        x_out = self.conv_with_max_pool(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.size_norm(out)
        return x_out + out



class ResNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 10
    ) -> None:
        super().__init__()

        self.in_channels = 3
        self.dilation = 1
        self.prep_layer = get_3x3_basic_conv_2d_layer(in_channels=self.in_channels, out_channels=64,stride=1, padding=1)
        self.layer1 = BasicResBlock(64,128,stride=1,padding=1)
        self.layer2 = get_custom_conv_with_max_pool_layer(128,256,stride=1, padding=1)
        self.layer3 = BasicResBlock(256,512,stride=1,padding=1)        
        self.layer4_max_pool = nn.MaxPool2d(kernel_size=4)
        self.fc = nn.Linear(512 , num_classes)

    def forward(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.prep_layer(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4_max_pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


