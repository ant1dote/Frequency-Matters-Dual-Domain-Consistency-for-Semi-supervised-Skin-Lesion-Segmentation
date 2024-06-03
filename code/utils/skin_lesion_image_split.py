import os
from pkgutil import extend_path
import json5
from tqdm import tqdm
import collections
from PIL import Image
import glob
import random

def get_files_with_extensions(folder_path, extension):
    file_list = []
    for file_name in glob.glob(os.path.join(folder_path, "*." +extension)):
        file_list.append(os.path.basename(file_name))
    return file_list

if __name__=='__main__':
    
    train_img_path = '/home/user/ISIC2018/ISIC2018_Task1-2_Training_Input'
    
    #val_img_path = '/home/user/ISIC2018/ISIC2018_Task1-2_Validation_Input'
    
    #test_img_path = '/home/user/ISIC2018/ISIC-2017_Test_v2_Data'

    img_files = get_files_with_extensions(train_img_path, 'jpg')

    sample_ratio = 1.00
    sample_num = int(len(img_files)*sample_ratio)
    sample_labeled_img = random.sample(img_files, sample_num)
    unlabeled_img = list(set(img_files) - set(sample_labeled_img))
    
    leak_list = []
    for labeled_img in sample_labeled_img:
        if labeled_img in unlabeled_img:
            leak_list.append(labeled_img)

    if sample_ratio == 0.1:
        labeled_list = r'10%_labeled_1.txt'
        unlabeled_list = r'10%_unlabeled_1.txt'
    elif sample_ratio == 1.00:
        labeled_list = r'FS_labeled.txt'
    elif sample_ratio == 0.01:
        labeled_list = r'1%_labeled_3.txt'
        unlabeled_list = r'1%_unlabeled_3.txt'
    elif sample_ratio == 0.2:
        labeled_list = r'20%_labeled.txt'
        unlabeled_list = r'20%_unlabeled.txt'
    
    txt_path = '/home/user/ISIC2018/'
    
    lab_list = open(txt_path+labeled_list, 'w')
    for labeled_img in sample_labeled_img:
        lab_list.write(labeled_img + '\n')
    lab_list.close()
    
    #unlab_list = open(txt_path+unlabeled_list, 'w')
    #for unlabeled_img in unlabeled_img:
    #    unlab_list.write(unlabeled_img + '\n')
    #unlab_list.close()    
    '''
    val_img = get_files_with_extensions(val_img_path, 'jpg')
    val_list = r'val_set.txt'
    val_list = open(txt_path + val_list, 'w')
    for img in val_img:
        val_list.write(img + '\n')
    val_list.close()

    test_img = get_files_with_extensions(test_img_path, 'jpg')
    test_list = r'test_set_1.txt'
    test_list = open(txt_path + test_list, 'w')
    for img in test_img:
        test_list.write(img + '\n')
    test_list.close()
    '''

    a=1
