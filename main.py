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
from nets.facenet import Facenet
from models import*
from fns_all import*
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

from nets.facenet_training import weights_init
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser()
#model and data locs
parser.add_argument('--train_meas_filenames',default='filenames/cls_train_meas_5000.txt')
parser.add_argument('--val_meas_filenames', default='filenames/cls_val_meas_5000.txt')
parser.add_argument('--train_orig_filenames', default='filenames/cls_train_orig_5000.txt')
parser.add_argument('--val_orig_filenames', default='filenames/cls_val_orig_5000.txt')
parser.add_argument('--architecture',default='UNET')
parser.add_argument('--modelRoot', default='flatnet_new')
parser.add_argument('--checkpoint', default='')
#lossweightage and gradientweightage
parser.add_argument('--wtp', default=1, type=float)
parser.add_argument('--wtmse', default=1, type=float)
# parser.add_argument('--wta', default=0.6, type=float)
parser.add_argument('--generatorLR', default=1e-4, type=float)
parser.add_argument('--facenetLR', default=1e-4, type=float)
parser.add_argument('--init', default='Transpose')
parser.add_argument('--num_classes', default=1000,type=int)
parser.add_argument('--numEpoch1', default=50,type=int)
parser.add_argument('--numEpoch2', default=100,type=int)
parser.add_argument('--valFreq', default=200,type=int)
parser.add_argument('--pretrain',dest='pretrain', action='store_true')
parser.set_defaults(pretrain=True)

opt = parser.parse_args()

device = torch.device("cuda:0")
datapath = '/home/hz/510*112/modelsave2/models/'
savedir = os.path.join(datapath, opt.modelRoot)
print(os.path)
print(savedir)
class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(os.path.join(save_dir, "log.txt"), "a+")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        
if not os.path.exists(savedir):
    os.mkdir(savedir)
sys.stdout = Logger(savedir)
print('======== Log ========')
print(datetime.now(timezone('Asia/Shanghai')))
print('\n')
print("Command ran:\n%s\n\n" % " ".join([x for x in sys.argv]))
print("Opt:")
pprint.pprint(vars(opt))
print("\n")
batchsize = 4
vla = float('inf')
k = 0
val_err = []
train_err = []
sys.stdout.flush()





if __name__ == '__main__':
    backbone="inception_resnetv1"
    Cuda=True
    facenet=Facenet(num_classes=opt.num_classes, backbone=backbone).to(device)
    weights_init(facenet)
    facenet_path="/home/hz/510*112/facenet_inception_resnetv1.pth"
    print('Loading weights into state dict...')
    
    facenet_dict = facenet.state_dict()
    pretrained_dict = torch.load(facenet_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(facenet_dict[k]) ==  np.shape(v)}
    facenet_dict.update(pretrained_dict)
    facenet.load_state_dict(facenet_dict)
    

    path_psf_init='/home/hz/510.png'
    PSF_init=cv2.imread(path_psf_init)
    PSF = PSF_init[:,:,0].astype(np.float32)
    tf = transforms.ToTensor()   ## 图像会转化为Python可以处理的格式
    PSF= tf(PSF).unsqueeze(0)
    PSF=PSF/255
    psf=PSF.to(device)
       
    gen = FlatNet(psf,1).to(device)
    
    
    gen_criterion = nn.MSELoss()
    facenet_criterion = nn.CosineEmbeddingLoss()
    
    
    ei = 0
    train_error = []
    val_error = []


    optim_gen = torch.optim.Adam(gen.parameters(), lr = opt.generatorLR)
    optim_facenet = torch.optim.Adam(facenet.parameters(), lr = opt.facenetLR)
    
    vla = float('inf')
    if opt.checkpoint:
        checkpoint = os.path.join(data, opt.checkpoint)
        ckpt = torch.load(checkpoint+'/latest.tar')
        optim_gen.load_state_dict(ckpt['optimizerG_state_dict'])
        optim_facenet.load_state_dict(ckpt['optimizerF_state_dict'])
        gen.load_state_dict(ckpt['gen_state_dict'])
        facenet.load_state_dict(ckpt['facenet_state_dict'])
        ei = ckpt['last_finished_epoch'] + 1
        val_error = ckpt['val_err']
        train_error = ckpt['train_err']
        vla = min(ckpt['val_err'])
        print('Loaded checkpoint from:'+checkpoint+'/latest.tar')

    for param_group in optim_gen.param_groups:
        genLR = param_group['lr']
        
    for param_group in optim_facenet.param_groups:
        faceLR = param_group['lr']   
        
    params_train = {'batch_size': 4,
            'shuffle': True,
            'num_workers': 16}

    params_val = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 4}
    
    train_loader = torch.utils.data.DataLoader(DatasetFromFilenames(opt.train_meas_filenames,opt.train_orig_filenames), **params_train)
    val_loader = torch.utils.data.DataLoader(DatasetFromFilenames(opt.val_meas_filenames,opt.val_orig_filenames), **params_val)


    wts = [opt.wtmse, opt.wtp]
    #训练unet
    for e in range(ei,opt.numEpoch1):
        since = time.time()
        sys.stdout.flush()
        train_error, val_error, vla, Xvalout = train_frozen_epoch(gen, facenet, wts, optim_gen, optim_facenet,train_loader, val_loader,gen_criterion,facenet_criterion,device, vla, e, savedir, train_error, val_error,sys.stdout,opt.valFreq)
        Xvalout = Xvalout.cpu()
        ims = Xvalout.detach().numpy()
        ims = ims[0, :, :, :]
        ims = np.swapaxes(np.swapaxes(ims,0,2),0,1)
        ims = (ims-np.min(ims))/(np.max(ims)-np.min(ims))
        skimage.io.imsave(savedir+'/latest.png', ims)

        dict_save = {
                'gen_state_dict': gen.state_dict(),
                'facenet_state_dict':facenet.state_dict(),
                'optimizerG_state_dict': optim_gen.state_dict(),
                'optimizerF_state_dict': optim_facenet.state_dict(),
                'train_err': train_error,
                'val_err': val_error,
                'last_finished_epoch': e,
                'opt': opt,
                'vla': vla}
        torch.save(dict_save, savedir+'/latest.tar')
#         savename = '/psf_epoch%d' % e
#         torch.save(gen.generator.conv1.state_dict(),savedir+savename)
        if e%2 == 0:
            genLR = genLR/2
            for param_group in optim_gen.param_groups:
                param_group['lr'] = genLR

        print('Saved latest')
        sys.stdout.flush()

        time_elapsed = time.time() - since
        print('Pretraining full Epoch complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    
    

