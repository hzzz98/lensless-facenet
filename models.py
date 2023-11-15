import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

device = torch.device("cuda:0")
# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)
def dilconv3x3(in_channels, out_channels, stride=1,dilation=2):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, dilation=dilation, padding=2, bias=False)



   

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch,momentum=0.99),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch,momentum=0.99),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class double_conv2(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3,stride=2, padding=1),
            nn.BatchNorm2d(out_ch,momentum=0.99),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch,momentum=0.99),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x    


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            double_conv2(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class upnocat(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(upnocat, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3,padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x


# In[7]:
def gasuss_noise(image, mean=0, var=0.5):
    

    noise = torch.normal(mean, var ** 0.5, image.shape)
    noise=noise.to(device)
    
    out = image + noise
    
    return out

def swish(x):
    return x * torch.sigmoid(x)

class convolution(nn.Module):
    def __init__(self,kernel,kernel_conv = 510):
        super(convolution, self).__init__()
        self.conv1 = nn.Conv2d(1,1,stride=1,kernel_size=kernel_conv,padding=509,bias=False)
        self.conv1.weight = nn.Parameter(kernel.flip(0,1), requires_grad=False)
#         self.conv2=nn.Conv2d(1,1,stride=1,kernel_size=2,padding=0,bias=False)
    def forward(self, x):
        x = self.conv1(x)

        return x

def wiener(x,PSF,K=100):
    
    A=torch.zeros(621,621)
    A[55:565,55:565]=PSF.flip(2,3)
    A=A.to(device)
  
    input_fft=torch.fft.fftn(x,dim=(2,3))

    PSF_fft=torch.fft.fft2(A)
    
    PSF_fft_conj=torch.conj(PSF_fft)
#     print(PSF_fft_conj.shape)
    PSF_fft_1=torch.div(PSF_fft_conj, (torch.mul(PSF_fft,PSF_fft_conj)+K))
#     print(PSF_fft_1.shape)
#     print(input_fft.shape)
    result=torch.mul(PSF_fft_1,input_fft) 
    result=torch.fft.ifftn(result,dim=(2,3))
    result=torch.fft.fftshift(result,dim=(2,3))
    result=torch.abs(result)
    result =result[:,:,256:368,256:368]
    maxx=torch.amax(result,axis=(2,3))
    maxx=(maxx.unsqueeze(2)).unsqueeze(3)
    maxx=maxx.repeat(1,1,112,112)
    result=torch.div(result,maxx)
    result=result.type(torch.float32)

    return result




    
    
class FlatNet(nn.Module):
    def __init__(self, psf,n_channels=1):
        super(FlatNet, self).__init__()
        
        self.PSF = nn.Parameter(psf)
        self.generator=convolution(self.PSF)
        self.inc = inconv(n_channels, 128)
        self.down1 = down(128, 256)
        self.down2 = down(256, 512)
        self.down3 = down(512, 1024)
        self.down4 = down(1024, 1024)
        self.up1 = up(2048, 512)
        self.up2 = up(1024, 256)
        self.up3 = up(512, 128)
        self.up4 = up(256, 128)
        self.outc = outconv(128, 1)
        self.bn=nn.BatchNorm2d(n_channels,momentum=0.99)
    def forward(self, Xinp):
        
        x,psf=self.generator(Xinp)
        x=gasuss_noise(Xinp, mean=0)
        xout= wiener(x,self.PSF,K=100)
#         print(self.PSF)

#         print(xout.shape)
        x = self.bn(xout)
#         print(x.shape)
        x1 = self.inc(x)
#         print(x1.shape)
        x2 = self.down1(x1)
#         print(x2.shape)
        x3 = self.down2(x2)
#         print(x3.shape)
        x4 = self.down3(x3)
#         print(x4.shape)
        x5 = self.down4(x4)
#         print(x5.shape)
        x = self.up1(x5, x4)
#         print(x.shape)
        x = self.up2(x, x3)
#         print(x.shape)
        x = self.up3(x, x2)
#         print(x.shape)
        x = self.up4(x, x1)
#         print(x.shape)
        x = self.outc(x)
#         print(torch.sigmoid(x).shape)
        

        return torch.sigmoid(x),xout





















