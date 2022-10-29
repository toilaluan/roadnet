import torch
import torch.nn as nn

# from .resnet import *


class SegDecoder(nn.Module):
    def __init__(self, in_channels = [64,128,256,512], inner_channels = 256, bias = False):
        super(SegDecoder, self).__init__()
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=False)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=False)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=False)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, bias=False)
        
        self.out5 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias = bias),
            nn.Upsample(scale_factor=8, mode='nearest')
        )
        self.out4 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias = bias),
            nn.Upsample(scale_factor=4, mode = 'nearest')
        )
        self.out3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding = 1, bias = bias),
            nn.Upsample(scale_factor=2, mode = 'nearest')
        )
        self.out2 = nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias = bias)
        self.binarize = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels // 4, 1, 3, padding=1, bias=bias),
            # nn.BatchNorm2d(inner_channels//4),
            # nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(inner_channels//4, inner_channels//4, 2, 2),
            # nn.Conv2d(inner_channels//4, 1, 1, padding=1, bias=bias),
            # nn.BatchNorm2d(inner_channels//4),
            # nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(inner_channels//4, 1, 2, 2),
            nn.Sigmoid())
    def forward(self, features):
        c2, c3, c4, c5 = features
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)
        # print(c4.shape)
        out4 = self.up5(in5) + in4
        out3 = self.up4(out4) + in3
        out2 = self.up3(out3) + in2
        
        p2 = self.out2(out2)
        p3 = self.out3(out3)
        p4 = self.out4(out4)
        p5 = self.out5(in5)
        
        fuse = torch.cat((p2,p3,p4,p5), 1)
        out = self.binarize(fuse)
        return out