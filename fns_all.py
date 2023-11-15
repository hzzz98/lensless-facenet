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
from models import*
import pdb



def validate(gen, facenet, wts, val_loader, gen_criterion,facenet_criterion, device):
    k = 0
    tloss = 0
    gen.eval()
    facenet.eval()
    with torch.no_grad():
        for X_val, Y_val in val_loader:
            batchsize = X_val.shape[0]
            X_val, Y_val = X_val.to(device), Y_val.to(device)
            X_valout,_ = gen(X_val)
            val1, val_outputs1 =facenet.forward_feature(X_valout)
            val2, val_outputs2 =facenet.forward_feature(Y_val)
            batchsize=np.array(batchsize)
            batchsize=torch.from_numpy(batchsize)
            batchsize=batchsize.to(device)
            if k == 5:
                op = X_valout
            tloss += wts[0]*(gen_criterion(Y_val, X_valout)+wts[1]*(facenet_criterion(val_outputs2,val_outputs1,batchsize)))
            k += 1
        tloss = tloss/len(val_loader)
    return op, tloss


def train_frozen_epoch(gen,facenet,wts, optim_gen,optim_facenet, train_loader, val_loader,gen_criterion,facenet_criterion,device, vla, e, savedir, train_error, val_error, ss,valFreq):
    i = 0
    
    for X_train, Y_train in train_loader:
        X_train, Y_train = X_train.to(device), Y_train.to(device)

        batchsize = X_train.shape[0]
                             
        for param in gen.parameters():
            param.requires_grad = True
            
        for param in facenet.parameters():
            param.requires_grad = False
                             
     
            
        optim_gen.zero_grad()
        gen.train()
        Xout,_ = gen(X_train)
        before_normalize1, outputs1 =facenet.forward_feature(Xout)
        before_normalize2, outputs2 =facenet.forward_feature(Y_train)
#         pdb.set_trace()
        batchsize=np.array(batchsize)
        batchsize=torch.from_numpy(batchsize)
        batchsize=batchsize.to(device)
                                           
        
        loss1 = wts[0]*gen_criterion(Y_train, Xout)
        loss2=  wts[1]*facenet_criterion(outputs2,outputs1,batchsize)
        loss=loss1+loss2
        
    
        loss.backward()
          
        optim_gen.step()

        
        
        train_error.append(loss.item())
        if i % valFreq == 0:
            Xvalout, vloss= validate(gen, facenet, wts, val_loader, gen_criterion,facenet_criterion, device)
            val_error.append(vloss)
            if vloss < vla:
                vla = vloss
                Xvalout = Xvalout.cpu()
                ims = Xvalout.detach().numpy()
                ims = ims[0, :, :, :]
                ims = np.swapaxes(np.swapaxes(ims,0,2),0,1)
                ims = (ims-np.min(ims))/(np.max(ims)-np.min(ims))
                skimage.io.imsave(savedir+'/best.png', ims)
                dict_save = {
                'gen_state_dict': gen.state_dict(),
                'optimizerG_state_dict': optim_gen.state_dict(),
                'train_err': train_error,
                'val_err': val_error,
                'last_finished_epoch': e}
                torch.save(dict_save, savedir+'/best.tar')
                print('Saved best')
        print('Epoch and Iterations::'+str(e)+','+str(i))
        print('Train and Val Loss:'+str(loss.item())+','+str(vloss))
        ss.flush()
        i += 1
    return train_error, val_error,  vla, Xvalout








