import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np







class convolution(nn.Module):
    def __init__(self,kernel,kernel_conv = 63):
        super(convolution, self).__init__()
        self.conv1 = nn.Conv2d(1,1,stride=1,kernel_size=kernel_conv,padding=62,bias=False)
        self.conv1.weight = nn.Parameter(kernel.flip(0,1), requires_grad=True)
#         self.conv2=nn.Conv2d(1,1,stride=1,kernel_size=2,padding=0,bias=False)
    def forward(self, x):
        x = self.conv1(x)

        return x
    
