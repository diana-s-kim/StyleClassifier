import torch
from torch import nn
from torchvision import models
import sys
import pandas as pd
import numpy as np


class BaseNet(nn.Module):
    def __init__(self,name='vgg16'):
        super().__init__()
        backbones={'vgg16_bn':models.vgg16_bn,
                   'vgg16':models.vgg16}
        self.name=name
        self.conv=nn.Sequential(*list(backbones[self.name](weights="IMAGENET1K_V1").children())[:-1])
        self.fc=nn.Sequential(*list(backbones[self.name]().children())[-1][:-1])
        self.basenet=nn.Sequential(self.conv,nn.Flatten(),self.fc)
        print(list(self.basenet.children()))
    def forward(self,x):
        x=self.basenet(x)
        return x #keep batch dim


#fc layer
class MLP(nn.Module):
    def __init__(self,layers,dropout,activations):
        super().__init__()
        fc=[nn.Flatten()]
        for layer,drop,relu in zip(layers,dropout,activations):
            linear_layer=nn.Linear(layer[0],layer[1])
            fc.append(linear_layer)
            if drop is not None:
                dropout_layer=nn.Dropout(p=drop)
                fc.append(dropout_layer)
            if relu is not None:
                relu_layer=nn.ReLU(inplace=True)
                fc.append(relu_layer)
        self.elements_layer=nn.Sequential(*fc)
    def forward(self,x):
        return self.elements_layer(x)
                    
