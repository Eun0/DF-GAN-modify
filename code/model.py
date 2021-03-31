import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict


class NetProjD(nn.Module):
    def __init__(self):
        super(NetProjD,self).__init__()
        self.feature_encoder = DISC_ENCODER()
        self.COND_DNET = DISC_LOGIT()

    def forward(self,imgs):
        # [bs,nef]
        _,features = self.feature_encoder(imgs)

        return features

class DISC_LOGIT(nn.Module):
    def __init__(self):
        super(DISC_LOGIT,self).__init__()
        self.cls = nn.Linear(256,1)

    def forward(self,features,sent_embs):
        # [bs]
        prob = self.cls(features)
        # [bs]
        match = torch.matmul(sent_embs,features.transpose(0,1))

        out = match + prob

        return out 


class DISC_ENCODER(nn.Module):
    def __init__(self):
        super(DISC_ENCODER,self).__init__()
        self.nef = 256
        self.nch = 32
        self.down_block1 = ResBlockDown(in_dim = 3         , out_dim = 1*self.nch)
        self.down_block2 = ResBlockDown(in_dim = 1*self.nch, out_dim = 2*self.nch)
        self.down_block3 = ResBlockDown(in_dim = 2*self.nch, out_dim = 4*self.nch)
        self.down_block4 = ResBlockDown(in_dim = 4*self.nch, out_dim = 8*self.nch)
        self.down_block5 = ResBlockDown(in_dim = 8*self.nch, out_dim = 8*self.nch)
        self.down_block6 = ResBlockDown(in_dim = 8*self.nch, out_dim = 16*self.nch)

        self.emb_features = nn.Linear(8*self.nch,self.nef)
        self.emb_code = ResBlock(in_dim = 16*self.nch, out_dim = self.nef)
        self.pool = nn.AvgPool2d(kernel_size=4,divisor_override=1)
        print('Use DISC encoder')

    def forward(self,x):

        x = self.down_block1(x)
        x = self.down_block2(x)
        x = self.down_block3(x)
        x = self.down_block4(x)
        image_features = x
        x = self.down_block5(x)
        x = self.down_block6(x)
        image_codes = self.emb_code(x)

        # [bs,8*nch,16,16] -> [bs,8*nch,16*16]
        image_features = image_features.view(image_features.size(0),image_features.size(1),-1)
        # [bs,8*nch,16*16] -> [bs,16*16,8*nch]
        image_features = image_features.transpose(1,2)
        # [bs,16*16,8*nch] -> [bs,16*16,nef]
        image_features = self.emb_features(image_features)
        # [bs,nef,16*16]
        image_features = image_features.permute(0,2,1)

        # Global Sum Pooling
        #[bs,nef,1,1]
        image_codes = self.pool(image_codes)
        #[bs,nef]
        image_codes = image_codes.view(image_codes.size(0),-1)

        return image_features,image_codes


def conv_nxn(in_planes, out_planes,n,bias=False,spec_norm=False):
    "1x1 convolution with padding"

    if n==1:
        padding = 0
    else:
        padding = 1

    conv = nn.Conv2d(in_planes,out_planes,kernel_size=n,stride=1,padding=padding,bias=bias)
    if spec_norm:
        conv = spectral_norm(conv)

    return conv


class ResBlock(nn.Module):
    def __init__(self,in_dim,out_dim,spec_norm=False):
        super(ResBlock,self).__init__()
        self.conv2d_res_1x1 = conv_nxn(in_planes=in_dim,out_planes=out_dim,n=1,spec_norm=spec_norm)
        self.conv2d_plain1_3x3 = conv_nxn(in_planes=in_dim,out_planes=out_dim,n=3,spec_norm=spec_norm)
        self.conv2d_plain2_3x3 = conv_nxn(in_planes=out_dim,out_planes=out_dim,n=3,spec_norm=spec_norm)
        
    def forward(self,x):
        x_res = self.conv2d_res_1x1(x)

        x_plain = nn.ReLU()(x)
        x_plain = self.conv2d_plain1_3x3(x_plain)
        x_plain = nn.ReLU()(x_plain)
        x_plain = self.conv2d_plain2_3x3(x_plain)

        out = x_res + x_plain

        return out
        
        
class ResBlockDown(ResBlock):
    def __init__(self,in_dim,out_dim,spec_norm=False):
        super(ResBlockDown,self).__init__(in_dim = in_dim, out_dim = out_dim, spec_norm = spec_norm)
        self.down = nn.AvgPool2d(kernel_size=2)

    def forward(self,x):
        x_res = self.conv2d_res_1x1(x)
        x_res = self.down(x_res)
        
        x_plain = nn.ReLU()(x)
        x_plain = self.conv2d_plain1_3x3(x_plain)
        x_plain = nn.ReLU()(x_plain)
        x_plain = self.conv2d_plain2_3x3(x_plain)
        x_plain = self.down(x_plain)

        out = x_res + x_plain
        return out



class NetG(nn.Module):
    def __init__(self, ngf=64, nz=100):
        super(NetG, self).__init__()
        self.ngf = ngf

        # layer1输入的是一个100x1x1的随机噪声, 输出尺寸(ngf*8)x4x4
        self.fc = nn.Linear(nz, ngf*8*4*4)
        self.block0 = G_Block(ngf * 8, ngf * 8)#4x4
        self.block1 = G_Block(ngf * 8, ngf * 8)#4x4
        self.block2 = G_Block(ngf * 8, ngf * 8)#8x8
        self.block3 = G_Block(ngf * 8, ngf * 8)#16x16
        self.block4 = G_Block(ngf * 8, ngf * 4)#32x32
        self.block5 = G_Block(ngf * 4, ngf * 2)#64x64
        self.block6 = G_Block(ngf * 2, ngf * 1)#128x128

        self.conv_img = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ngf, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x, c):

        out = self.fc(x)
        out = out.view(x.size(0), 8*self.ngf, 4, 4)
        out = self.block0(out,c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block1(out,c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block2(out,c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block3(out,c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block4(out,c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block5(out,c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block6(out,c)

        out = self.conv_img(out)

        return out



class G_Block(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(G_Block, self).__init__()

        self.learnable_sc = in_ch != out_ch 
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.affine0 = affine(in_ch)
        self.affine1 = affine(in_ch)
        self.affine2 = affine(out_ch)
        self.affine3 = affine(out_ch)
        self.gamma = nn.Parameter(torch.zeros(1))
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch,out_ch, 1, stride=1, padding=0)

    def forward(self, x, y=None):
        return self.shortcut(x) + self.gamma * self.residual(x, y)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, x, y=None):
        h = self.affine0(x, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        h = self.affine1(h, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        h = self.c1(h)
        
        h = self.affine2(h, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        h = self.affine3(h, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        return self.c2(h)



class affine(nn.Module):

    def __init__(self, num_features):
        super(affine, self).__init__()

        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(256, 256)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(256, num_features)),
            ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(256, 256)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(256, num_features)),
            ]))
        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y=None):

        weight = self.fc_gamma(y)
        bias = self.fc_beta(y)        

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        size = x.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * x + bias


class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf

        self.joint_conv = nn.Sequential(
            nn.Conv2d(ndf * 16+256, ndf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf * 2, 1, 4, 1, 0, bias=False),
        )

    def forward(self, out, y):
        
        y = y.view(-1, 256, 1, 1)
        y = y.repeat(1, 1, 4, 4)
        h_c_code = torch.cat((out, y), 1)
        out = self.joint_conv(h_c_code)
        return out





# 定义鉴别器网络D
class NetD(nn.Module):
    def __init__(self, ndf):
        super(NetD, self).__init__()

        self.conv_img = nn.Conv2d(3, ndf, 3, 1, 1)#128
        self.block0 = resD(ndf * 1, ndf * 2)#64
        self.block1 = resD(ndf * 2, ndf * 4)#32
        self.block2 = resD(ndf * 4, ndf * 8)#16
        self.block3 = resD(ndf * 8, ndf * 16)#8
        self.block4 = resD(ndf * 16, ndf * 16)#4
        self.block5 = resD(ndf * 16, ndf * 16)#4

        self.COND_DNET = D_GET_LOGITS(ndf)

    def forward(self,x):

        out = self.conv_img(x)
        out = self.block0(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)

        return out




class resD(nn.Module):
    def __init__(self, fin, fout, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.learned_shortcut = (fin != fout)
        self.conv_r = nn.Sequential(
            nn.Conv2d(fin, fout, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(fout, fout, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_s = nn.Conv2d(fin,fout, 1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, c=None):
        return self.shortcut(x)+self.gamma*self.residual(x)

    def shortcut(self, x):
        if self.learned_shortcut:
            x = self.conv_s(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x

    def residual(self, x):
        return self.conv_r(x)












