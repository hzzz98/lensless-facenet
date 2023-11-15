import numpy as np
import skimage
import torch
from PIL import Image
import torchvision
import torch.nn.functional as F
from io import BytesIO
import cv2
import pdb





class DatasetFromFilenames:

    def __init__(self, filenames_loc_meas):
        self.filenames_meas = filenames_loc_meas
        self.paths_meas = get_paths(self.filenames_meas)
        self.mask=nn.Parameter(mask)
        self.gen=convolution(self.mask)
        self.num_im = len(self.paths_meas)
        self.totensor = torchvision.transforms.ToTensor()
#         self.resize = torchvision.transforms.Resize((256,256))
        

    def __len__(self):
        return len(self.paths_meas)

    def __getitem__(self, index):
        # obtain the image paths
#         print(index)
        
        meas_path = self.paths_meas[index % self.num_im]
        label=meas_path.split('/')[6]
        # load images (grayscale for direct inference)
        

        meas = Image.open(meas_path)
        meas = np.asarray(meas).astype(np.float32) 
        meas=meas[:,:,:]
        meas=meas/255
        

        meas = self.totensor(meas)
        
       


        return meas,label

def get_paths(fname):
    paths = []
    with open(fname, 'r') as f:
        for line in f:
            temp = str(line).strip()
            paths.append(temp)
    return paths

