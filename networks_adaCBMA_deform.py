# adaptive CBMA + Deformable conv2

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import deform_conv_v2 as deformConv


import sys
import math

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Ada_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Ada_BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

        Weight1 = [conv3x3(planes, planes),
                    nn.Sigmoid()]
        Weight2 = [conv3x3(planes, planes),
                    nn.Sigmoid()]
        Weight3 = [conv3x3(planes, planes),
                    nn.Sigmoid()]

        FC = [nn.Linear(planes, planes, bias=False),
                  nn.ReLU(True),
                  nn.Linear(planes, planes, bias=False),
                  nn.ReLU(True)]

        self.Weight1 = nn.Sequential(*Weight1)
        self.Weight2 = nn.Sequential(*Weight2)
        self.FC = nn.Sequential(*FC)          

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out_c = self.ca(out) * out # 通道注意力
        out_s = self.sa(out) * out # 空间注意力
        ########
        wight1 = self.Weight1(out_c) # 注意力机制加权分支
        out_c1 = out_c*wight1
        #######
        wight2 = self.Weight2(out) # 注意力机制加权分支
        out_s2 = out_s*wight2
        ######
        wight3 = torch.cat([wight1.view(wight1.shape[0],-1).unsqueeze(0), wight1.view(wight1.shape[0],-1).unsqueeze(0)], 0)
        wight3_temp = torch.max(wight3,0)[0]
    
        wight3 = 1-wight3_temp.view(wight1.shape)
        out_temp = out_c*wight3
        out_s3 = self.sa(out_temp) * out_temp
        
        out = out_c1 + out_s2 + out_s3

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

##################################################

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, adaNorm = False,skip=True,deformable=True,dcn_ksize=1):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.adaNorm = adaNorm
        self.skip = skip
        self.deformable = deformable 

        DownBlock = []
        if deformable :
            DownBlock += [
                      deformConv.DeformConv2d(input_nc,ngf, kernel_size=dcn_ksize, stride=1, padding=dcn_ksize//2),
                      nn.InstanceNorm2d(ngf),
                      nn.ReLU(True)]
        else:
            DownBlock += [nn.ReflectionPad2d(3),
                      nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=False),
                      nn.InstanceNorm2d(ngf),
                      nn.ReLU(True)]


        # Down-Sampling
        DownBlock1 = []
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock1 += [nn.ReflectionPad2d(1),
                          nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0, bias=False),
                          nn.InstanceNorm2d(ngf * mult * 2),
                          nn.ReLU(True)]

        # Down-Sampling Bottleneck
        DownBlock2 = []
        mult = 2**n_downsampling
        for i in range(n_blocks):
            DownBlock2 += [ResnetBlock(ngf * mult, use_bias=False)]
            

        # Class Activation Map
        self.gap_fc = nn.Linear(ngf * mult, 1, bias=False)
        self.gmp_fc = nn.Linear(ngf * mult, 1, bias=False)
        self.conv1x1 = nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(True)

        # Gamma, Beta block

        FC = [nn.Linear(ngf * mult, ngf * mult, bias=False),
              nn.ReLU(True),
              nn.Linear(ngf * mult, ngf * mult, bias=False),
              nn.ReLU(True)]

        self.gamma = nn.Linear(ngf * mult, ngf * mult, bias=False)
        self.beta = nn.Linear(ngf * mult, ngf * mult, bias=False)

        # Up-Sampling Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i+1), ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            UpBlock2 += [nn.Upsample(scale_factor=2, mode='nearest'),
                        nn.Conv2d(2*ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=False),
                        #deformConv.DeformConv2d(2*ngf * mult, int(ngf * mult / 2), kernel_size=1, stride=1, padding=0),
                        ILN(int(ngf * mult / 2)),
                        nn.ReLU(True)]
        
                            
        if deformable :
            UpBlock_final = []
            UpBlock_final += [deformConv.DeformConv2d(ngf, output_nc, kernel_size=dcn_ksize, stride=1, padding=dcn_ksize//2),
                            nn.Tanh()]
        else:
            UpBlock_final = []
            UpBlock_final += [nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=3, bias=False),
                            nn.Tanh()]

        self.DownBlock = nn.Sequential(*DownBlock)
        self.DownBlock1_1 = nn.Sequential(*DownBlock1[0:4])
        self.DownBlock1_2 = nn.Sequential(*DownBlock1[4:-1]) 
        self.DownBlock2 = nn.Sequential(*DownBlock2)

        #self.UpBlock2 = nn.Sequential(*UpBlock2)
        self.UpBlock2_1 = nn.Sequential(*UpBlock2[0:4])
        self.UpBlock2_2 = nn.Sequential(*UpBlock2[4:-1])
        self.UpBlock2_final = nn.Sequential(*UpBlock_final)


        self.FC = nn.Sequential(*FC)
        self.Ada_BasicBlock = Ada_BasicBlock(ngf * mult *2, ngf * mult*2)

            
            

    def forward(self, input):
        x = self.DownBlock(input)
        x_d1 = self.DownBlock1_1(x)
        x_d2 = self.DownBlock1_2(x_d1)
        x = self.DownBlock2(x_d2)

        # adaptive CBMA
        x = self.Ada_BasicBlock(x) 
        

        if self.adaNorm:
            x_ = torch.nn.functional.adaptive_avg_pool2d(x, 1)
            x_ = self.FC(x_.view(x_.shape[0], -1))
            gamma, beta = self.gamma(x_), self.beta(x_) # 1*128 map 1*128

            for i in range(self.n_blocks):
                x = getattr(self, 'UpBlock1_' + str(i+1))(x, gamma, beta)
        else:
            x = self.DownBlock2(x)
        ###############
        #out = self.UpBlock2(x)
        x = torch.cat([x,x_d2],1)
        x = self.UpBlock2_1(x)

        x = torch.cat([x, x_d1], 1)
        x = self.UpBlock2_2(x)

        out = self.UpBlock2_final(x)
        #################
        
        if self.skip:
            out = out + input   

        return out     


class ResnetBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetAdaILNBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = adaILN(dim)
        self.relu1 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x


class adaILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.9)

    def forward(self, input, gamma, beta):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)

        return out


class ILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)

        return out

