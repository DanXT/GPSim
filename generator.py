# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 16:45:57 2019

@author: xtan
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=0, batch_norm=True):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    # create a sequence of transpose + optional batch norm layers
    layers = []
    transpose_conv_layer = nn.ConvTranspose2d(in_channels, out_channels, 
                                              kernel_size, stride, padding, output_padding, bias=False)
    # append transpose convolutional layer
    layers.append(transpose_conv_layer)
    
    if batch_norm:
        # append batchnorm layer
        layers.append(nn.BatchNorm2d(out_channels))
        
    return nn.Sequential(*layers)



class Generator(nn.Module):
    
    def __init__(self, z_size, conv_dim=32):
        super(Generator, self).__init__()

        # complete init function
        
        self.conv_dim = conv_dim
        
        # first, fully-connected layer
        self.fc = nn.Linear(z_size, conv_dim*4*2*2)

        # transpose conv layers
        self.t_conv1 = deconv(conv_dim*4, conv_dim*2, 4, output_padding=1)
        self.t_conv2 = deconv(conv_dim*2, conv_dim, 4, output_padding=0)
        self.t_conv3 = deconv(conv_dim, 1, 4, output_padding=1, batch_norm=False)
        

    def forward(self, x):
        # fully-connected + reshape 
        out = self.fc(x)
        out = out.view(-1, self.conv_dim*4, 2, 2) # (batch_size, depth, 4, 4)
        
        # hidden transpose conv layers + relu
        out = F.relu(self.t_conv1(out))
        out = F.relu(self.t_conv2(out))
        
        # last layer + tanh activation
        out = self.t_conv3(out)
        out = F.tanh(out)
        
        return out