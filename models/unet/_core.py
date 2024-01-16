# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:09:35 2023

@author: Gavin
"""

import torch, math

#import torch.functional as F

from torch import nn
from models import SqueezeExcitation, ConvBlock, SEFusion

# based on implementation from https://github.com/TsukamotoShuchi/RCNN/blob/master/rcnnblock.py
class RCL2D(nn.Module):
    
    def __init__(self, channels, steps=3):
        super().__init__()
        self.conv = nn.Conv3d(
            channels, 
            channels, 
            kernel_size=(1, 3, 3), 
            stride=(1, 1, 1), 
            padding=(0, 1, 1), 
            bias=False
        )
        self.bn = nn.ModuleList([nn.BatchNorm3d(channels) for i in range(steps)])
        self.relu = nn.ReLU(inplace=True)
        self.steps = steps

        self.shortcut = nn.Conv3d(
            channels, 
            channels, 
            kernel_size=(1, 3, 3), 
            stride=(1, 1, 1), 
            padding=(0, 1, 1), bias=False
        )



    def forward(self, x):
        rx = x
        for i in range(self.steps):
            if i == 0:
                z = self.conv(x)
            else:
                z = self.conv(x) + self.shortcut(rx)
            x = self.relu(z)
            x = self.bn[i](x)
        return x
    
    

class RR2DBlock(nn.Module):
    
    def __init__(
        self, 
        channels
    ):
        super().__init__()
        
        self.rcl1 = RCL2D(channels)
        self.bn1 = nn.BatchNorm3d(channels)
        self.a1 = nn.ReLU(inplace=True)
        
        self.rcl2 = RCL2D(channels)
        self.bn2 = nn.BatchNorm3d(channels)
        self.a2 = nn.ReLU(inplace=True)
        
        
        
    def forward(self, x):
        out1 = self.a1(self.bn1(self.rcl1(x)))
        out2 = self.rcl2(out1)
        
        out3 = self.bn2(out2 + x)
        out4 = self.a2(out3)
        
        return out4



class RREL2D(nn.Module):
    
    def __init__(self, in_channels, out_channels, enc_ratio):
        super().__init__()
        
        layers = []
        if enc_ratio >= 2:
            mp = nn.MaxPool3d(
                kernel_size=(1, enc_ratio, enc_ratio), 
                stride=(1, enc_ratio, enc_ratio)
            )
            
            layers.append(mp)
        elif enc_ratio < 1:
            raise ValueError('enc_ratio must be an integer and 1 or larger')
        
        cn = nn.Conv3d(
            in_channels, 
            out_channels, 
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            stride=(1, 1, 1)
        )
        bn = nn.BatchNorm3d(out_channels)
        a = nn.ReLU(inplace=True)
        b = RR2DBlock(out_channels)
        
        layers.extend((cn, bn, a, b))
        
        self.features = nn.Sequential(*layers)
        
    
    
    def forward(self, x):
        return self.features(x)
    
    
    
class RRDL2D(nn.Module):
    
    def __init__(self, in_channels, out_channels, dec_ratio):
        super().__init__()
        
        if dec_ratio == 3:
            self.dcn = nn.ConvTranspose3d(
                in_channels, 
                out_channels,
                kernel_size=(1, 3, 3),
                stride=(1, 3, 3),
                padding=(0, 0, 0)
            )
        elif dec_ratio == 2:
            self.dcn = nn.ConvTranspose3d(
                in_channels, 
                out_channels,
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
                output_padding=(0, 1, 1)
            )
        else:
            raise ValueError('dec_ratio other than 2 or 3 is not supported')
        
        self.a_1 = nn.ReLU(inplace=True)
        self.bn_1 = nn.BatchNorm3d(out_channels)
        self.cn = nn.Conv3d(
            out_channels * 2,
            out_channels, 
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            stride=(1, 1, 1)
        )
        self.a_2 = nn.ReLU(inplace=True)
        self.bn_2 = nn.BatchNorm3d(out_channels)
        self.b = RR2DBlock(out_channels)
        
        
        
    def forward(self, x, skip):
        if len(x.shape) == 5:
            dim = 1
        elif len(x.shape) == 4:
            dim = 0
        
        out = torch.cat((self.bn_1(self.a_1(self.dcn(x))), skip), dim=dim)
        out = self.b(self.bn_2(self.a_2(self.cn(out))))

        return out



class RCL3D(RCL2D):
    
    def __init__(self, channels, steps=3):
        super().__init__(channels, steps=steps)
        
        self.conv = nn.Conv3d(
            channels, 
            channels, 
            kernel_size=(3, 3, 3), 
            stride=(1, 1, 1), 
            padding=(1, 1, 1), 
            bias=False
        )

        self.shortcut = nn.Conv3d(
            channels, 
            channels, 
            kernel_size=(3, 3, 3), 
            stride=(1, 1, 1), 
            padding=(1, 1, 1), 
            bias=False
        )
    
    
    
class RR3DBlock(RR2DBlock):
    
    def __init__(self, channels):
        super().__init__(channels)
        
        self.rcl1 = RCL3D(channels)
        self.rcl2 = RCL3D(channels)
    
    
    
class RREL3D(RREL2D):
    
    def __init__(self, in_channels, out_channels, enc_ratio):
        super().__init__(in_channels, out_channels, enc_ratio)
        
        layers = []
        if enc_ratio >= 2:
            mp = nn.MaxPool3d(
                kernel_size=(1, enc_ratio, enc_ratio), 
                stride=(1, enc_ratio, enc_ratio)
            )
            
            layers.append(mp)
        elif enc_ratio < 1:
            raise ValueError('enc_ratio must be an integer and 1 or larger')
        
        cn = nn.Conv3d(
            in_channels, 
            out_channels, 
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            stride=(1, 1, 1)
        )
        bn = nn.BatchNorm3d(out_channels)
        a = nn.ReLU(inplace=True)
        b = RR3DBlock(out_channels)
        
        layers.extend((cn, bn, a, b))
        
        self.features = nn.Sequential(*layers)
    
    
    
class RRDL3D(RRDL2D):
    
    def __init__(self, in_channels, out_channels, dec_ratio):
        super().__init__(in_channels, out_channels, dec_ratio)
        
        if dec_ratio == 3:
            self.dcn = nn.ConvTranspose3d(
                in_channels, 
                out_channels,
                kernel_size=(1, 3, 3),
                stride=(1, 3, 3),
                padding=(0, 0, 0)
            )
        elif dec_ratio == 2:
            self.dcn = nn.ConvTranspose3d(
                in_channels, 
                out_channels,
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
                output_padding=(0, 1, 1)
            )
        else:
            raise ValueError('dec_ratio other than 2 or 3 is not supported')
        
        self.cn = nn.Conv3d(
            out_channels * 2,
            out_channels, 
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            stride=(1, 1, 1)
        )
        
        self.b = RR3DBlock(out_channels)
        


class AttentionBlock2D(nn.Module):
    
    def __init__(self, channels):
        super().__init__()
        
        self.rb1 = self.__residual_block(channels)
        self.rb2 = self.__residual_block(channels)
        self.rb3 = self.__residual_block(channels, final=True)
        
        self.skip = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(channels),
        )
        
        self.aout = nn.Sigmoid()
        
        
        
    def forward(self, x):
        out1 = self.rb1(x) + x
        out2 = self.rb2(out1) + out1
        out3 = self.aout(self.rb3(out2) + self.skip(out2))
        
        return out3
        
        
        
    def __residual_block(self, channels, final=False):
        layers = [
            nn.Conv3d(channels, channels, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                channels, 
                channels, 
                kernel_size=(1, 3, 3),
                stride=(1, 1, 1),
                padding=(0, 1, 1)
            ),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(channels)    
        ]
        
        if not final:
            layers.append(nn.ReLU(inplace=True))
            
        return nn.Sequential(*layers)
    
    
    
class AttentionBlock3D(AttentionBlock2D):
    
    def __residual_block(self, channels, final=False):
        layers = [
            nn.Conv3d(channels, channels, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                channels, 
                channels, 
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding=(1, 1, 1)
            ),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(channels)    
        ]
        
        if not final:
            layers.append(nn.ReLU(inplace=True))
            
        return nn.Sequential(*layers)
    
    
    
class MultiAttentionBlock2D(nn.Module):
    
    def __init__(self, channels):
        super().__init__()
        
        self.spatial = AttentionBlock2D(1)
        self.channel = SqueezeExcitation(channels, channels)
        self.conv = ConvBlock(channels, channels)
        self.activation = nn.Sigmoid()
    
    
    
    def forward(self, x):
        if len(x.shape) == 4:
            dim = 0
        elif len(x.shape) == 5:
            dim = 1
        
        s_out = self.spatial(torch.mean(x, dim=dim, keepdim=True))
        c_out = self.channel(x)
        
        out = s_out * c_out
        out = self.conv(out)
        out = self.activation(out)
        
        return out
        
        
        
class MultiAttentionBlock3D(MultiAttentionBlock2D):
    
    def __init__(self, channels):
        super().__init__(channels)
        
        self.spatial = AttentionBlock3D(1)
    
    
    
class MAMRCL2D(RCL2D):
    
    def __init__(self, channels, steps=2):
        super().__init__(channels, steps)
        
        self.conv = MAMBlock2D(channels)
        
    
    
class MAMBlock2D(nn.Module):
    
    def __init__(self, channels):
        super().__init__()
        
        self.res = Res2NetBlock2D(channels, channels)
        self.ed = EncDecBlock2D(channels)
        self.att = AttentionBlock2D(channels)



    def forward(self, x):
        residual = self.res(x)

        encdec = self.ed(residual)
        att = self.att(residual)      

        out = residual + (residual * att) + encdec
        
        return out
    


class EncDecBlock2D(nn.Module):
    
    def __init__(self, channels):
        super().__init__()
        
        self.l1 = nn.Sequential(
            nn.Conv3d(
                channels, 
                channels, 
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                stride=(1, 2, 2),
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(channels)
        )
        
        self.l2 = nn.Sequential(
            nn.Conv3d(
                channels, 
                channels, 
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                stride=1,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(channels)
        )
        
        self.l3 = nn.Sequential(
            nn.Conv3d(
                channels, 
                channels, 
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                stride=1,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(channels)
        )
        
        self.l4 = nn.Sequential(
            nn.ConvTranspose3d(
                channels,
                channels,
                kernel_size=(1, 2, 2),
                stride=(1, 2, 2),
                padding=0
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(channels)
        )
    
    
    
    def forward(self, x):
        residual = self.l1(x)
        residual = self.l3(self.l2(residual)) + residual
        
        return self.l4(residual)
                


        
# Implementation based on: https://github.com/Res2Net/Res2Net-PretrainedModels/blob/master/res2net.py
class Res2NetBlock2D(nn.Module):
    expansion = 1

    def __init__(self,
        inplanes, 
        planes, 
        stride=1,
        baseWidth=26,
        scale=4,
        stype='normal'
    ):
        super().__init__()

        width = int(math.floor(planes * (baseWidth/64.0)))
        
        self.conv1 = nn.Conv3d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(width*scale)
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
          
        if stype == 'stage':
            self.pool = nn.AvgPool3d(
                kernel_size=(1, 3, 3), 
                stride=stride, 
                padding=(0, 1, 1)
            )
            
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv3d(
              width,
              width,
              kernel_size=(1, 3, 3),
              stride=stride,
              padding=(0, 1, 1),
              bias=False
          ))
          
          bns.append(nn.BatchNorm3d(width))
          
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv3d(width*scale, planes*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes*self.expansion)

        self.relu = nn.ReLU(inplace=True)
        
        self.stype = stype
        self.scale = scale
        self.width  = width



    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
            
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
          
        
        if self.scale != 1 and self.stype=='normal':
          out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
          out = torch.cat((out, self.pool(spx[self.nums])),1)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += residual
        out = self.relu(out)

        return out
    
    
    
class MARMEL2D(RREL2D):
    
    def __init__(self, in_channels, out_channels, enc_ratio):
        super().__init__(in_channels, out_channels, enc_ratio)
        
        layers = []
        if enc_ratio >= 2:
            mp = nn.MaxPool3d(
                kernel_size=(1, enc_ratio, enc_ratio), 
                stride=(1, enc_ratio, enc_ratio)
            )
            
            layers.append(mp)
        elif enc_ratio < 1:
            raise ValueError('enc_ratio must be an integer and 1 or larger')
        
        cn = nn.Conv3d(
            in_channels, 
            out_channels, 
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            stride=(1, 1, 1)
        )
        bn = nn.BatchNorm3d(out_channels)
        a = nn.ReLU(inplace=True)
        b = MAMRCL2D(out_channels)
        
        layers.extend((cn, bn, a, b))
        
        self.features = nn.Sequential(*layers)
        


class MAMRCL3D(RCL3D):
    
    def __init__(self, channels, steps=2):
        super().__init__(channels, steps)
        
        self.conv = MAMBlock3D(channels)
    
    
    
class MAMBlock3D(MAMBlock2D):
    
    def __init__(self, channels):
        super().__init__(channels)
        
        self.res = Res2NetBlock3D(channels, channels)
        self.ed = EncDecBlock3D(channels)
        self.att = AttentionBlock3D(channels)
    


class EncDecBlock3D(EncDecBlock2D):
    
    def __init__(self, channels):
        super().__init__(channels)
        
        self.l1 = nn.Sequential(
            nn.Conv3d(
                channels, 
                channels, 
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
                stride=(1, 2, 2),
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(channels)
        )
        
        self.l2 = nn.Sequential(
            nn.Conv3d(
                channels, 
                channels, 
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
                stride=1,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(channels)
        )
        
        self.l3 = nn.Sequential(
            nn.Conv3d(
                channels, 
                channels, 
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
                stride=1,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(channels)
        )
        
        self.l4 = nn.Sequential(
            nn.ConvTranspose3d(
                channels,
                channels,
                kernel_size=(1, 2, 2),
                stride=(1, 2, 2),
                padding=0
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(channels)
        )
    

                  
class Res2NetBlock3D(Res2NetBlock2D):

    def __init__(self,
        inplanes, 
        planes, 
        stride=1,
        baseWidth=26,
        scale=4,
        stype='normal'
    ):
        super().__init__(
            inplanes, 
            planes, 
            stride=1,
            baseWidth=26,
            scale=4,
            stype='normal'
        )

        width = int(math.floor(planes * (baseWidth/64.0)))
        
        self.conv1 = nn.Conv3d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(width*scale)
            
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv3d(
              width,
              width,
              kernel_size=(3, 3, 3),
              stride=stride,
              padding=(1, 1, 1),
              bias=False
          ))
          
          bns.append(nn.BatchNorm3d(width))
          
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv3d(width*scale, planes*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes*self.expansion)
    
    
    
class MARMEL3D(MARMEL2D):
    
    def __init__(self, in_channels, out_channels, enc_ratio):
        super().__init__(in_channels, out_channels, enc_ratio)
        
        layers = []
        if enc_ratio >= 2:
            mp = nn.MaxPool3d(
                kernel_size=(1, enc_ratio, enc_ratio), 
                stride=(1, enc_ratio, enc_ratio)
            )
            
            layers.append(mp)
        elif enc_ratio < 1:
            raise ValueError('enc_ratio must be an integer and 1 or larger')
        
        cn = nn.Conv3d(
            in_channels, 
            out_channels, 
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            stride=(1, 1, 1)
        )
        bn = nn.BatchNorm3d(out_channels)
        a = nn.ReLU(inplace=True)
        b = MAMRCL3D(out_channels)
        
        layers.extend((cn, bn, a, b))
        
        self.features = nn.Sequential(*layers)
        
        
        
class SERes2NetBlock2D(nn.Module):
    
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        
        self.res = Res2NetBlock2D(in_channels, out_channels, **kwargs)
        self.se = SqueezeExcitation(out_channels, out_channels)
        
        
        
    def forward(self, x):
        out = self.se(self.res(x))
        out = out + x
        
        return out
    
    
    
class SERes2NetBlock3D(SERes2NetBlock2D):
    
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        
        self.res = Res2NetBlock3D(in_channels, out_channels, **kwargs)
        
        
        
class MARBlock2D(MAMBlock2D):
    
    def __init__(self, channels):
        super().__init__(channels)
        
        self.res = SERes2NetBlock2D(channels, channels)
        self.ed = EncDecBlock2D(channels)
        self.att = MultiAttentionBlock2D(channels)



class MARBlock3D(MAMBlock3D):
    
    def __init__(self, channels):
        super().__init__(channels)
        
        self.res = SERes2NetBlock3D(channels, channels)
        self.ed = EncDecBlock3D(channels)
        self.att = MultiAttentionBlock3D(channels)
        
        
        
class MARRCL2D(nn.Module):
    
    def __init__(self, channels, steps=3):
        super().__init__()
        
        self.conv = MARBlock2D(channels)
        self.bn = nn.ModuleList([nn.BatchNorm3d(channels) for i in range(steps)])
        self.relu = nn.ReLU(inplace=True)
        self.steps = steps
        self.shortcut = SERes2NetBlock2D(channels, channels)
        self.fuse = SEFusion(channels, 2)



    def forward(self, x):
        rx = x
        for i in range(self.steps):
            if i == 0:
                z = self.conv(x)
            else:
                
                z = self.fuse(self.conv(x), self.shortcut(rx))
            x = self.relu(z)
            x = self.bn[i](x)
        return x
    
    
    
class MARRCL3D(MARRCL2D):
    
    def __init__(self, channels, steps=3):
        super().__init__(channels, steps)
        
        self.conv = MARBlock3D(channels)
        self.shortcut = SERes2NetBlock3D(channels, channels)
        
        
        
class MAREL2D(RREL2D):
    
    def __init__(self, in_channels, out_channels, enc_ratio):
        super().__init__(in_channels, out_channels, enc_ratio)
        
        layers = []
        if enc_ratio >= 2:
            mp = nn.MaxPool3d(
                kernel_size=(1, enc_ratio, enc_ratio), 
                stride=(1, enc_ratio, enc_ratio)
            )
            
            layers.append(mp)
        elif enc_ratio < 1:
            raise ValueError('enc_ratio must be an integer and 1 or larger')
        
        cn = nn.Conv3d(
            in_channels, 
            out_channels, 
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            stride=(1, 1, 1)
        )
        bn = nn.BatchNorm3d(out_channels)
        a = nn.ReLU(inplace=True)
        b = MARRCL2D(out_channels)
        
        layers.extend((cn, bn, a, b))
        
        self.features = nn.Sequential(*layers)
        
        
        
class MAREL3D(RREL3D):
    
    def __init__(self, in_channels, out_channels, enc_ratio):
        super().__init__(in_channels, out_channels, enc_ratio)
        
        layers = []
        if enc_ratio >= 2:
            mp = nn.MaxPool3d(
                kernel_size=(1, enc_ratio, enc_ratio), 
                stride=(1, enc_ratio, enc_ratio)
            )
            
            layers.append(mp)
        elif enc_ratio < 1:
            raise ValueError('enc_ratio must be an integer and 1 or larger')
        
        cn = nn.Conv3d(
            in_channels, 
            out_channels, 
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            stride=(1, 1, 1)
        )
        bn = nn.BatchNorm3d(out_channels)
        a = nn.ReLU(inplace=True)
        b = MARRCL3D(out_channels)
        
        layers.extend((cn, bn, a, b))
        
        self.features = nn.Sequential(*layers)
        
        