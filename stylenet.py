import torch
from torch import nn
import neuralnet

class StyleNet(nn.Module):
    def __init__(self,name='vgg16',mlp=[[4096,1024],[1024,512],[512,20]],dropout=[0.5,0.5,0.5],activations=['relu','relu','relu']):
        super().__init__()
        self.net=neuralnet.BaseNet(name=name)
        self.add_fc=neuralnet.MLP(mlp,dropout,activations)
        self.logit=nn.Sequential(self.net, self.add_fc)#neuralnet.BaseNet(name=name,drop=1),neuralnet.MLP(mlp,dropout,activations))
    def forward(self,x):
        return self.logit(x)
        
