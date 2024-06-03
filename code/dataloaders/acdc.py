from dataloaders.transform import random_rot_flip, random_rotate, blur, obtain_cutmix_box, random_rot_flip_rgb

from copy import deepcopy
import h5py
import math
import numpy as np
import os
from PIL import Image, ImageEnhance
import random
from scipy.ndimage.interpolation import zoom
from scipy import ndimage
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms
import pywt
from math import sqrt


def hflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask

def vflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    return img, mask

def rotate(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.ROTATE_90)
        mask = mask.transpose(Image.ROTATE_90)
    return img, mask


class ISICDataset_DIFF(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open(self.root+'/val_set.txt', 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        
        id = self.ids[item].split('.')[0]+'.h5'
        
        if 'train' in self.mode:
            h5f = h5py.File(self.root + "/train_h5/{}".format(id), 'r')
        
        if self.mode == 'val':
            h5f = h5py.File(self.root + "/val_h5/{}".format(id), 'r')
            img, mask = h5f['image'][:], h5f['label'][:]
            return img, mask

        img, mask = h5f['image'][:], h5f['label'][:]
        img = Image.fromarray((img.transpose((1, 2, 0)) * 255).astype(np.uint8))
        mask = Image.fromarray(mask * 255)

        if random.random() > 0.5:
            img, mask = hflip(img, mask)
        if random.random() > 0.5:
            img, mask = vflip(img, mask)
        if random.random() > 0.5:
            img, mask = rotate(img, mask)
       
        if self.mode == 'train_l':
            img, mask = np.asarray(img) / 255.0, np.asarray(mask) // 255
            img = img.transpose((2, 0, 1))
            return torch.from_numpy(img).float(), torch.from_numpy(np.array(mask)).long()
        
        
        img_s1, img_s2 = deepcopy(img), deepcopy(img)
        img = torch.from_numpy(np.array(img)).permute((2, 0, 1)).float() / 255.0
        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
            
        
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(self.size, p=0.5)
        
        img_s1_copy = deepcopy(img_s1)
        img_s1_copy = np.asarray(img_s1_copy)
        
        img_s1 = torch.from_numpy(np.array(img_s1)).permute((2, 0, 1)).float() / 255.0

        cutmix_box2 = obtain_cutmix_box(self.size, p=0.5)

        
        if random.random() < 0.8:
            
            img_s2 = np.asarray(img_s2).transpose((2, 0, 1)).astype(np.float32) / 255.0
            coeffs_s2 = pywt.dwt2(img_s2, 'haar')
            ll_s2, details_s2 = coeffs_s2
            
            min_ll_s2, max_ll_s2 = np.min(ll_s2), np.max(ll_s2)
            min_img, max_img = 0, 255
            ll_s2 = (((ll_s2 - min_ll_s2) / (max_ll_s2 - min_ll_s2)) * max_img).astype(np.uint8) 
            ll_s2 = Image.fromarray(ll_s2.astype(np.uint8).transpose((1, 2, 0)), mode='RGB')
            
            
            ll_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(ll_s2)
            
            min_, max_ = np.min(ll_s2), np.max(ll_s2)
            ll_s2 = min_ll_s2 + ((ll_s2 - min_)/(max_ - min_)) * (max_ll_s2 - min_ll_s2)
            ll_s2 = ll_s2.transpose((2, 0, 1))
            
            
            list_details_s2 = list(details_s2)
            
            #apply gaussian kernel to blur the horizontal high-frequency component
            
            hf_min, hf_max = np.min(list_details_s2[0]), np.max(list_details_s2[0])
            list_details_s2[0] = (((list_details_s2[0] - hf_min) / (hf_max - hf_min)) * 255).astype(np.uint8)
            list_details_s2[0] = Image.fromarray(list_details_s2[0].transpose((1,2,0)))           
            list_details_s2[0] = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(list_details_s2[0])       
            list_details_s2[0] = np.array(list_details_s2[0]).astype(np.float32)
            min_blur_hf1, max_blur_hf1 = np.min(list_details_s2[0]), np.max(list_details_s2[0])
            list_details_s2[0] = hf_min + ((list_details_s2[0] - min_blur_hf1) / (max_blur_hf1 - min_blur_hf1)) * (hf_max - hf_min)
            list_details_s2[0] = list_details_s2[0].transpose((2, 0, 1))
          
            
            # apply gaussian kernel to blur the vertical high-frequency component
            
            hf_min, hf_max = np.min(list_details_s2[1]), np.max(list_details_s2[1])
            list_details_s2[1] = (((list_details_s2[1] - hf_min) / (hf_max - hf_min)) * 255).astype(np.uint8)
            list_details_s2[1] = Image.fromarray(list_details_s2[1].astype(np.uint8).transpose((1,2,0)), mode='RGB')
            list_details_s2[1] = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(list_details_s2[1])
            
            list_details_s2[1] = np.array(list_details_s2[1]).astype(np.float32)
            min_blur_hf1, max_blur_hf1 = np.min(list_details_s2[1]), np.max(list_details_s2[1])
            list_details_s2[1] = hf_min + ((list_details_s2[1] - min_blur_hf1) / (max_blur_hf1 - min_blur_hf1)) * (hf_max - hf_min)
            list_details_s2[1] = list_details_s2[1].transpose((2, 0, 1))
            
           

            hf_min, hf_max = np.min(list_details_s2[2]), np.max(list_details_s2[2])
            list_details_s2[2] = (((list_details_s2[2] - hf_min) / (hf_max - hf_min)) * 255).astype(np.uint8)
            list_details_s2[2] = Image.fromarray(list_details_s2[2].astype(np.uint8).transpose((1,2,0)), mode='RGB')
            
            list_details_s2[2] = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(list_details_s2[2])
            list_details_s2[2] = np.array(list_details_s2[2]).astype(np.float32)
            min_blur_hf1, max_blur_hf1 = np.min(list_details_s2[2]), np.max(list_details_s2[2])
            list_details_s2[2] = hf_min + ((list_details_s2[2] - min_blur_hf1) / (max_blur_hf1 - min_blur_hf1)) * (hf_max - hf_min)
            list_details_s2[2] = list_details_s2[2].transpose((2, 0, 1))
    
           
            details_s2 = tuple(list_details_s2)
            img_s2 = pywt.idwt2((ll_s2, details_s2), wavelet='haar')
            
            img_min, img_max = np.min(img_s2), np.max(img_s2)
            img_s2_png = (((img_s2 - img_min) / (img_max - img_min)) * 255).astype(np.uint8)
            img_s2_png = Image.fromarray(img_s2_png.transpose((1, 2, 0)).astype(np.uint8), mode='RGB')
            
            img_s2_png_copy = np.asarray(img_s2_png)
            
            img_diff_2 = abs(img_s2_png_copy-img_s1_copy)
            

            img_add_2 = img_s2_png_copy + img_diff_2
            
            img_add_2 = img_add_2.transpose((2, 0, 1)) / 255.0
            img_add_2 = torch.from_numpy(img_add_2).float()
            img_s2 = torch.from_numpy(np.array(img_s2)).float() / 255.0

            # output img_diff2:
            img_diff_2 = img_diff_2.transpose((2, 0, 1)) / 255.0
            img_diff_2 = torch.from_numpy(img_diff_2).float()

            return img, img_s1, img_add_2, cutmix_box1, cutmix_box2
        else:
            img_s2 = torch.from_numpy(np.array(img_s2)).permute((2, 0, 1)).float() / 255.0
            return img, img_s1, img_s2, cutmix_box1, cutmix_box2
        
    def __len__(self):
        return len(self.ids)

class HAMDataset_DIFF(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open(self.root+'/val_set_3.txt', 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        
        id = self.ids[item].split('.')[0]+'.h5'
        
        if 'train' in self.mode:
            h5f = h5py.File(self.root + "/train_3/{}".format(id), 'r')
        
        if self.mode == 'val':
            h5f = h5py.File(self.root + "/val_3/{}".format(id), 'r')
            img, mask = h5f['image'][:], h5f['label'][:]
            return img, mask

        img, mask = h5f['image'][:], h5f['label'][:]
        img = Image.fromarray((img.transpose((1, 2, 0)) * 255).astype(np.uint8))
        mask = Image.fromarray(mask * 255)

        if random.random() > 0.5:
            img, mask = hflip(img, mask)
        if random.random() > 0.5:
            img, mask = vflip(img, mask)
        if random.random() > 0.5:
            img, mask = rotate(img, mask)
       
        if self.mode == 'train_l':
            img, mask = np.asarray(img) / 255.0, np.asarray(mask) // 255
            img = img.transpose((2, 0, 1))
            return torch.from_numpy(img).float(), torch.from_numpy(np.array(mask)).long()
        
        
        img_s1, img_s2 = deepcopy(img), deepcopy(img)
        img = torch.from_numpy(np.array(img)).permute((2, 0, 1)).float() / 255.0
        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
            
        
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(self.size, p=0.5)
        
        img_s1_copy = deepcopy(img_s1)
        img_s1_copy = np.asarray(img_s1_copy)
        
        
        img_s1 = torch.from_numpy(np.array(img_s1)).permute((2, 0, 1)).float() / 255.0        
        cutmix_box2 = obtain_cutmix_box(self.size, p=0.5)
                
        if random.random() < 1.0:
            
            img_s2 = np.asarray(img_s2).transpose((2, 0, 1)).astype(np.float32) / 255.0
            coeffs_s2 = pywt.dwt2(img_s2, 'haar')
            ll_s2, details_s2 = coeffs_s2
            
           
            list_details_s2 = list(details_s2)
            
            #apply gaussian kernel to blur the horizontal high-frequency component
            
            hf_min, hf_max = np.min(list_details_s2[0]), np.max(list_details_s2[0])
            list_details_s2[0] = (((list_details_s2[0] - hf_min) / (hf_max - hf_min)) * 255).astype(np.uint8)
            list_details_s2[0] = Image.fromarray(list_details_s2[0].transpose((1,2,0)))
            
            list_details_s2[0] = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(list_details_s2[0])
            
            list_details_s2[0] = np.array(list_details_s2[0]).astype(np.float32)
            min_blur_hf1, max_blur_hf1 = np.min(list_details_s2[0]), np.max(list_details_s2[0])
            list_details_s2[0] = hf_min + ((list_details_s2[0] - min_blur_hf1) / (max_blur_hf1 - min_blur_hf1)) * (hf_max - hf_min)
            list_details_s2[0] = list_details_s2[0].transpose((2, 0, 1))
                      
            # apply gaussian kernel to blur the vertical high-frequency component
            
            hf_min, hf_max = np.min(list_details_s2[1]), np.max(list_details_s2[1])
            list_details_s2[1] = (((list_details_s2[1] - hf_min) / (hf_max - hf_min)) * 255).astype(np.uint8)
            list_details_s2[1] = Image.fromarray(list_details_s2[1].astype(np.uint8).transpose((1,2,0)), mode='RGB')
            list_details_s2[1] = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(list_details_s2[1])
            
            list_details_s2[1] = np.array(list_details_s2[1]).astype(np.float32)
            min_blur_hf1, max_blur_hf1 = np.min(list_details_s2[1]), np.max(list_details_s2[1])
            list_details_s2[1] = hf_min + ((list_details_s2[1] - min_blur_hf1) / (max_blur_hf1 - min_blur_hf1)) * (hf_max - hf_min)
            list_details_s2[1] = list_details_s2[1].transpose((2, 0, 1))
            
            
                        
            hf_min, hf_max = np.min(list_details_s2[2]), np.max(list_details_s2[2])
            list_details_s2[2] = (((list_details_s2[2] - hf_min) / (hf_max - hf_min)) * 255).astype(np.uint8)
            list_details_s2[2] = Image.fromarray(list_details_s2[2].astype(np.uint8).transpose((1,2,0)), mode='RGB')
            
            list_details_s2[2] = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(list_details_s2[2])
            list_details_s2[2] = np.array(list_details_s2[2]).astype(np.float32)
            min_blur_hf1, max_blur_hf1 = np.min(list_details_s2[2]), np.max(list_details_s2[2])
            list_details_s2[2] = hf_min + ((list_details_s2[2] - min_blur_hf1) / (max_blur_hf1 - min_blur_hf1)) * (hf_max - hf_min)
            list_details_s2[2] = list_details_s2[2].transpose((2, 0, 1))
    
            
            details_s2 = tuple(list_details_s2)
            img_s2 = pywt.idwt2((ll_s2, details_s2), wavelet='haar')
            
            img_min, img_max = np.min(img_s2), np.max(img_s2)
            img_s2_png = (((img_s2 - img_min) / (img_max - img_min)) * 255).astype(np.uint8)
            img_s2_png = Image.fromarray(img_s2_png.transpose((1, 2, 0)).astype(np.uint8), mode='RGB')
           
            img_s2_png_copy = np.asarray(img_s2_png)
            
            img_diff_2 = abs(img_s2_png_copy-img_s1_copy)
            
            img_add_2 = img_s2_png_copy + img_diff_2
            
            img_add_2 = img_add_2.transpose((2, 0, 1)) / 255.0
            img_add_2 = torch.from_numpy(img_add_2).float()
            img_s2 = torch.from_numpy(np.array(img_s2)).float() / 255.0

            
            img_diff_2 = img_diff_2.transpose((2, 0, 1)) / 255.0
            img_diff_2 = torch.from_numpy(img_diff_2).float()

            
            return img, img_s1, img_add_2, cutmix_box1, cutmix_box2
        else:
            img_s2 = torch.from_numpy(np.array(img_s2)).permute((2, 0, 1)).float() / 255.0
            return img, img_s1, img_s2, cutmix_box1, cutmix_box2
        
    def __len__(self):
        return len(self.ids)


