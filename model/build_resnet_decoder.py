# from turtle import forward
from .resnet import *
from .decoders import *
import torch.nn as nn
import torch

class ResnetSeg(nn.Module):
    def __init__(self):
        super(ResnetSeg, self).__init__()
        self.backbones = resnet18()
        self.decoders = SegDecoder()
        
    def forward(self, x):
        features = self.backbones(x)
        
        outputs = self.decoders(features)
        
        return outputs
    
if __name__ == '__main__':
    model = ResnetSeg()
    model.eval()
    x = torch.zeros((1,3,448,448))
    print(model(x).shape)