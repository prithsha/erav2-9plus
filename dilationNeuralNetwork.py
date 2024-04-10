import torch
from torch import nn
import torch.nn.functional as F


def get_basic_conv_2d_layer(in_channels, out_channels, stride=1, dilation = 1, drop_out=0.0, padding = 0, kernel_size = 3):
    conv_layer = nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=out_channels, dilation = dilation,                  
                  kernel_size=kernel_size, stride=stride, padding=padding, bias=False),            
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Dropout(drop_out)
    )

    return conv_layer



class DepthWiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthWiseSeparableConv, self).__init__()
        self.depthWiseConv2d = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                         groups=in_channels, stride=stride, padding=padding, bias=False)
        self.pointWiseConv2d = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthWiseConv2d(x)
        x = self.pointWiseConv2d(x)
        return x

def get_depth_wise_conv_layer(in_channels, out_channels, stride=1, drop_out=0.0, padding = 0, kernel_size = 3):
    conv_layer = nn.Sequential(
        DepthWiseSeparableConv(in_channels=in_channels,out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding),            
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Dropout(drop_out)
    )

    return conv_layer

def get_conv_layer(in_channels, out_channels, stride=1,  dilation = 1, drop_out = 0.1, padding=1, kernel_size=3, use_depth_wise_conv = False):
  if(use_depth_wise_conv):
        return get_depth_wise_conv_layer(in_channels=in_channels, out_channels=out_channels, stride=stride,
                                        drop_out=drop_out, padding=padding, kernel_size=kernel_size)
  else:
        return get_basic_conv_2d_layer(in_channels=in_channels, out_channels=out_channels, stride=stride, dilation=dilation,
                                        drop_out=drop_out, padding=padding, kernel_size=kernel_size)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, last_layer_stride=1,  drop_out = 0.1, use_dilation_at_last_layer = False,
                 use_depth_wise_conv_layer = False):
        super(BasicBlock, self).__init__()

        # Applying dilation or stride only on last layer
        dilation = 1
        if(use_dilation_at_last_layer and use_depth_wise_conv_layer == False):
            dilation = 2

        self.conv_layer_first = get_conv_layer(in_channels=in_channels, out_channels=out_channels, stride=1,
                                                drop_out= drop_out, padding=0,
                                                use_depth_wise_conv=use_depth_wise_conv_layer)

        self.conv_layer_2 = get_conv_layer(in_channels=out_channels, out_channels=out_channels, stride=1,
                                                drop_out= drop_out, padding=1, 
                                                use_depth_wise_conv=use_depth_wise_conv_layer )
        
        # self.conv_layer_3 = get_conv_layer(in_channels=out_channels, out_channels=out_channels, stride=1,
        #                                         drop_out= drop_out, padding=1,
        #                                         use_depth_wise_conv=use_depth_wise_conv_layer )

        if(use_dilation_at_last_layer):
            self.conv_layer_last = get_conv_layer(in_channels=out_channels, out_channels=out_channels, stride=1,
                                                dilation=dilation, drop_out= drop_out, padding=1,
                                                use_depth_wise_conv=False)
        else:
            self.conv_layer_last = get_conv_layer(in_channels=out_channels, out_channels=out_channels, stride=last_layer_stride,
                                                drop_out= drop_out, padding=1,
                                                use_depth_wise_conv=use_depth_wise_conv_layer )
            
        

    def forward(self, x):
        out = self.conv_layer_first(x)
        out = self.conv_layer_2(out)
        # out = self.conv_layer_3(out)
        out = self.conv_layer_last(out)
        return out


class DilationNeuralNetwork(nn.Module):

    def __init__(self, drop_out = 0.1):
        super().__init__()
        self.drop_out = drop_out


        self.conv_block1 = BasicBlock(in_channels=3,out_channels=64, drop_out=drop_out, last_layer_stride=2)    
        self.transition_layer1 = get_basic_conv_2d_layer(in_channels=64, out_channels=32, drop_out= drop_out,
                                                            padding=0, kernel_size=1) # RF-7, O-15
        

        # Dilation calculation : RF = (Rin - 1) * D + K + 1
        self.conv_block2 = BasicBlock(in_channels=32,out_channels=64, drop_out=drop_out, use_dilation_at_last_layer=True)        
        self.transition_layer2 = get_basic_conv_2d_layer(in_channels=64, out_channels=32, drop_out= drop_out,
                                                            padding=0, kernel_size=1)  # RF-32, O-11

        
        self.conv_block3 = BasicBlock(in_channels=32,out_channels=64, drop_out=drop_out, last_layer_stride=2, use_depth_wise_conv_layer=True)
        self.transition_layer3 = get_basic_conv_2d_layer(in_channels=64, out_channels=32, drop_out= drop_out,
                                                            padding=0, kernel_size=1) # RF-38, O-5
        
        
        self.conv_block4 = BasicBlock(in_channels=32,out_channels=32, drop_out=drop_out, last_layer_stride=2, use_depth_wise_conv_layer=True)
                                                                # RF-50, O-2


        # Output block
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 10)



    def forward(self, x):
        x = self.conv_block1(x)
        x = self.transition_layer1(x)

        x = self.conv_block2(x)
        x = self.transition_layer2(x)

        x = self.conv_block3(x) 
        x = self.transition_layer3(x)

        x = self.conv_block4(x)
        
        x = self.gap(x)
        x = x.view(-1,32)
        x = self.fc1(x)
        x = self.fc2(x)
        
             
        return F.log_softmax(x, dim=1)
