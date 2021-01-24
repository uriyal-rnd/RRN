import os
import torch.utils.data as data
import torch
import numpy as np
from PIL import Image, ImageOps
import random
from bicubic import imresize
from Gaussian_downsample import gaussian_downsample
from utils import our_normalize

def load_img(image_path, scale, rgb_type=True):
    HR = []
    #for img_num in range(7):
    for img_num in range(4): #changed to 4 to fit in gpu
        if rgb_type:
            GT_temp = modcrop(Image.open(os.path.join(image_path,'im{}.png'.format(img_num+1))).convert('RGB'), scale)
        else:
            #GT_temp = modcrop(Image.open(os.path.join(image_path,'im{}.png'.format(img_num+1))), scale)
            GT_temp = modcrop(Image.open(os.path.join(image_path)), scale)
            GT_temp = our_normalize(GT_temp)
        HR.append(GT_temp)
    return HR

def modcrop(img,scale):
    (ih, iw) = img.size
    ih = ih - ( ih % scale)
    iw = iw - ( iw % scale)
    img = img.crop((0,0,ih,iw))
    return img

def train_process(GH, flip_h=True, rot=True, converse=True): 
    if random.random() < 0.5 and flip_h: 
        GH = [ImageOps.flip(LR) for LR in GH]
    if rot:
        if random.random() < 0.5:
            GH = [ImageOps.mirror(LR) for LR in GH]
    return GH

class DataloadFromFolder(data.Dataset): # load train dataset
    def __init__(self, image_dir, scale, data_augmentation, file_list, transform, rgb_type):
        super(DataloadFromFolder, self).__init__()
        alist = [line.rstrip() for line in open(os.path.join(image_dir,file_list))] 
        self.image_filenames = [os.path.join(image_dir,x) for x in alist] 
        self.scale = scale
        self.transform = transform # To_tensor
        self.data_augmentation = data_augmentation # flip and rotate
        self.rgb_type = rgb_type
    def __getitem__(self, index):
        GT = load_img(self.image_filenames[index], self.scale, self.rgb_type)
        GT = train_process(GT) # input: list (contain PIL), target: PIL
        GT = [np.asarray(HR) for HR in GT]  # PIL -> numpy # input: list (contatin numpy: [H,W,C])
        GT = np.asarray(GT) # numpy, [T,H,W,C]
        if self.scale == 4:
            if self.rgb_type:
                GT = np.lib.pad(GT, pad_width=(
                (0, 0), (2 * self.scale, 2 * self.scale), (2 * self.scale, 2 * self.scale), (0, 0)), mode='reflect')
            else:
                GT = np.lib.pad(GT,
                                pad_width=((0, 0), (2 * self.scale, 2 * self.scale), (2 * self.scale, 2 * self.scale)),
                                mode='reflect')
        if self.rgb_type:
            t, h, w, c = GT.shape
            GT = GT.transpose(1, 2, 3, 0).reshape(h, w, -1)  # numpy, [H',W',CT]
        else:
            t,h,w = GT.shape
            c = 1
            GT = GT.transpose(1, 2, 0).reshape(h, w, -1)  # numpy, [H',W',CT]

        if self.transform:
            GT = self.transform(GT) # Tensor, [CT',H',W']
        GT = GT.view(c,t,h,w) # Tensor, [C,T,H,W]
        LR = gaussian_downsample(GT, self.scale)
        LR = torch.cat((LR[:,1:2,:,:], LR), dim=1)
        return LR, GT

    def __len__(self):
        return len(self.image_filenames) 

