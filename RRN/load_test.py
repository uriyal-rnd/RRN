import os
import torch.utils.data as data
import torch
import numpy as np
from PIL import Image, ImageOps
import random
from Gaussian_downsample import gaussian_downsample
from bicubic import imresize
from utils import our_normalize

def modcrop(img,scale):
    (iw, ih) = img.size
    ih = ih - (ih % scale)
    iw = iw - (iw % scale)
    img = img.crop((0,0,iw,ih))
    return img

class DataloadFromFolderTest(data.Dataset): # load test dataset
    def __init__(self, image_dir, scale, scene_name, transform, rgb_type=True):
        super(DataloadFromFolderTest, self).__init__()
        alist = os.listdir(os.path.join(image_dir, scene_name))
        alist.sort()
        self.image_filenames = [os.path.join(image_dir, scene_name, x) for x in alist] 
        self.L = len(alist)
        self.scale = scale
        self.transform = transform # To_tensor
        self.rgb_type = rgb_type
    def __getitem__(self, index):
        target = []
        for i in range(self.L):
            if self.rgb_type:
                GT_temp = modcrop(Image.open(self.image_filenames[i]).convert('RGB'), self.scale)
            else:
                GT_temp = modcrop(Image.open(self.image_filenames[i]), self.scale)
                GT_temp = our_normalize(GT_temp)
            target.append(GT_temp)
        target = [np.asarray(HR) for HR in target] 
        target = np.asarray(target)
        if self.scale == 4:
            if self.rgb_type:
                target = np.lib.pad(target, pad_width=((0,0), (2*self.scale,2*self.scale), (2*self.scale,2*self.scale), (0,0)), mode='reflect')
            else:
                target = np.lib.pad(target, pad_width=((0,0), (2*self.scale,2*self.scale), (2*self.scale,2*self.scale)), mode='reflect')
        if self.rgb_type:
            t, h, w, c = target.shape
            target = target.transpose(1, 2, 3, 0).reshape(h, w, -1)  # numpy, [H',W',CT']
        else:
            t, h, w = target.shape
            c = 1
            target = target.transpose(1, 2, 0).reshape(h, w, -1)  # numpy, [H',W',CT']

        if self.transform:
            target = self.transform(target) # Tensor, [CT',H',W']
        target = target.view(c,t,h,w)
        LR = gaussian_downsample(target, self.scale) # [c,t,h,w]
        LR = torch.cat((LR[:,1:2,:,:], LR), dim=1)
        return LR, target
        
    def __len__(self):
        return 1 

