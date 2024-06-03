import os
from pkgutil import extend_path
import json5
from tqdm import tqdm
import collections
from PIL import Image
import glob
import random
import numpy as np
import PIL.Image as Image
import h5py

if __name__=='__main__':
    
    root_path = '/home/user/ISIC2017'
    
    train_img_path = root_path + '/ISIC-2017_Training_Data'
    train_lab_path = root_path + '/ISIC-2017_Training_Part1_GroundTruth'
    
    val_img_path = root_path + '/ISIC-2017_Validation_Data'
    val_lab_path = root_path + '/ISIC-2017_Validation_Part1_GroundTruth'
    
    test_img_path = root_path + '/ISIC-2017_Test_v2_Data'
    test_lab_path = root_path + '/ISIC-2017_Test_v2_Part1_GroundTruth'
    
    img_size = 256

    with open(root_path + '/10%_labeled_1.txt', 'r') as f1:
        train_lsample_list = f1.readlines()
    train_lsample_list = [item.replace('\n', '') for item in train_lsample_list]

    with open(root_path + '/10%_unlabeled_1.txt', 'r') as f1:
        train_usample_list = f1.readlines()
    train_usample_list = [item.replace('\n', '') for item in train_usample_list]

    with open(root_path + '/val_set_1.txt', 'r') as f1:
        val_sample_list = f1.readlines()
    val_sample_list = [item.replace('\n', '') for item in val_sample_list]

    with open(root_path + '/test_set_1.txt', 'r') as f1:
        test_sample_list = f1.readlines()
    test_sample_list = [item.replace('\n', '') for item in test_sample_list]
    
    train_h5_save_path = root_path + '/train_h5'
    val_h5_save_path = root_path + '/val_h5'
    test_h5_save_path = root_path + '/test_h5'
    save_path = [train_h5_save_path, val_h5_save_path, test_h5_save_path]

    for path in save_path:
        if not os.path.exists(path):
            os.makedirs(path)

    for file in tqdm(train_lsample_list):
        img = Image.open(train_img_path+'/'+file)
        label_name = file.split('.')[0] + "_segmentation.png"
        label  = Image.open(train_lab_path+'/'+label_name)
        img, label = img.resize((img_size, img_size), Image.BICUBIC), label.resize((img_size, img_size), Image.NEAREST) 
        img = np.asarray(img).astype(np.float32).transpose((2, 0, 1)) / 255.0
        label = np.asarray(label).astype(np.uint8) // 255
        
        h5_file_name = file.split('.')[0]+'.h5'
        h5_file = h5py.File(train_h5_save_path + '/' + h5_file_name, 'w')
        h5_file['image'], h5_file['label'] = img, label
        h5_file.close()

    #h5 = h5py.File(train_h5_save_path+'/'+ train_lsample_list[200].split('.')[0]+'.h5','r')
    #img = h5['image'][:]
    #label = h5['label'][:]
    
    for file in tqdm(train_usample_list):
        img = Image.open(train_img_path+'/'+file)
        label_name = file.split('.')[0] + "_segmentation.png"
        label  = Image.open(train_lab_path+'/'+label_name)
        img, label = img.resize((img_size, img_size), Image.BICUBIC), label.resize((img_size, img_size), Image.NEAREST) 
        img = np.asarray(img).astype(np.float32).transpose((2, 0, 1)) / 255.0
        label = np.asarray(label).astype(np.uint8) // 255
        
        h5_file_name = file.split('.')[0]+'.h5'
        h5_file = h5py.File(train_h5_save_path + '/' + h5_file_name, 'w')
        h5_file['image'], h5_file['label'] = img, label
        h5_file.close()

    for file in tqdm(val_sample_list):
        img = Image.open(val_img_path+'/'+file)
        label_name = file.split('.')[0] + "_segmentation.png"
        label  = Image.open(val_lab_path+'/'+label_name)
        img, label = img.resize((img_size, img_size), Image.BICUBIC), label.resize((img_size, img_size), Image.NEAREST) 
        img = np.asarray(img).astype(np.float32).transpose((2, 0, 1)) / 255.0
        label = np.asarray(label).astype(np.uint8) // 255
        
        h5_file_name = file.split('.')[0]+'.h5'
        h5_file = h5py.File(val_h5_save_path + '/' + h5_file_name, 'w')
        h5_file['image'], h5_file['label'] = img, label
        h5_file.close()

    for file in tqdm(test_sample_list):
        img = Image.open(test_img_path+'/'+file)
        label_name = file.split('.')[0] + "_segmentation.png"
        label  = Image.open(test_lab_path+'/'+label_name)
        img, label = img.resize((img_size, img_size), Image.BICUBIC), label.resize((img_size, img_size), Image.NEAREST) 
        img = np.asarray(img).astype(np.float32).transpose((2, 0, 1)) / 255.0
        label = np.asarray(label).astype(np.uint8) // 255
        
        h5_file_name = file.split('.')[0]+'.h5'
        h5_file = h5py.File(test_h5_save_path + '/' + h5_file_name, 'w')
        h5_file['image'], h5_file['label'] = img, label
        h5_file.close()

    a=1