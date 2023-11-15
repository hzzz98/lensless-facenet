import torch
import scipy.io as sio
import numpy as np
import os
from skimage.color import rgb2gray
import skimage.io
import random
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import resize as rsz
import torch.optim as optim
import os

from models_1 import*
from fns_all_1 import*
from dataloader_psf import*
import argparse
from torch.utils import data
import torchvision.transforms as transforms
import skimage.transform
import copy
import sys
import pprint
from datetime import datetime
from pytz import timezone
import time
import cv2
from tqdm import tqdm
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser()
#model and data locs

parser.add_argument('--val_meas_filenames', default='filenames/full.txt')
parser.add_argument('--val_orig_filenames', default='filenames/full.txt')
opt = parser.parse_args()
class convolution(nn.Module):
    def __init__(self,kernel,kernel_conv = 510):
        super(convolution, self).__init__()
        self.conv1 = nn.Conv2d(1,1,stride=1,kernel_size=kernel_conv,padding=509,bias=False)
        self.conv1.weight = nn.Parameter(kernel.flip(0,1), requires_grad=False)
#         self.conv2=nn.Conv2d(1,1,stride=1,kernel_size=2,padding=0,bias=False)
    def forward(self, x):
        x = self.conv1(x)
        psf=self.conv1.weight
        return x
def test(test_loader, model):
    # switch to evaluate mode
    model.eval()


    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p,label,name) in pbar:
        
        with torch.no_grad():
            data_a= data_a.type(torch.FloatTensor)
            if cuda:
                data_a= data_a.cuda()
            data_a= Variable(data_a)
            path='/home/hz/510*112/init_01.png'
            PSF_init=cv2.imread(path)
            PSF = PSF_init[:,:,0].astype(np.float32)
            tf = transforms.ToTensor()  
            PSF= tf(PSF).unsqueeze(0)
            PSF=PSF/255
            PSF=PSF.to(device)
            generator=convolution(kernel=PSF)
            X_val=generator.forward(data_a)
            Xvalout,xout = model(X_val)
            Xvalout=Xvalout.cpu()
            ims = Xvalout.numpy()
            ims = ims[0, :, :, :]
            ims = np.swapaxes(np.swapaxes(ims,0,2),0,1)
            ims = (ims-np.min(ims))/(np.max(ims)-np.min(ims))
            savedir='/media/pi/data4/hz/112*510-psf'
            a=label[0]
            label=a.split('/')[5]
#             label=label[0]
            name=a.split('/')[6]

            skimage.io.imsave(savedir+'/'+label+'/'+name, ims)
            
        
        if batch_idx % log_interval == 0:
            pbar.set_description('Test Epoch: [{}/{} ({:.0f}%)]'.format(
                batch_idx * batch_size, len(test_loader.dataset),
                100. * batch_idx / len(test_loader)))
    
   
if __name__ == "__main__":
    pthfile = '/home/hz/510*112/modelsave3/models/flatnet_new/latest.tar' #faster_rcnn_ckpt.pth
    net = torch.load(pthfile,map_location=torch.device('cuda'))

    cuda = True
    batch_size = 1
    log_interval = 1
    params_test = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 4}
    test_loader = torch.utils.data.DataLoader(DatasetFromFilenames(opt.val_meas_filenames,opt.val_orig_filenames),**params_test)
    PSF=torch.randn(1,1,510,510)
    model = FlatNet(PSF,1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(net["gen_state_dict"], strict=False)
    model  = model.eval()
    model = model.cuda()

    test(test_loader, model)
    