import argparse
import os
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
import PIL.Image as Image
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/user/HAM10000', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='Dual_Domain_Match_HAM10000_1%_RE1', help='experiment_name')
parser.add_argument('--model', type=str, default='unets_noa', help='model_name')
parser.add_argument('--num_classes', type=int,  default=2, help='output channel of network')
parser.add_argument('--labelnum', type=int, default=20, help='labeled data')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--dataset', type = str, default = 'HAM10000')

def calculate_metric_percase(pred, gt):
    
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)

    return dice, jc, hd95, asd


def test_single_volume(case, net, test_save_path, FLAGS):
    
    h5f = h5py.File(FLAGS.root_path + "/test_3/{}".format(case.split('.')[0]+'.h5'), 'r')
    slice, label = h5f['image'][:], h5f['label'][:]
    prediction = np.zeros_like(label)
    
    x, y = slice.shape[1], slice.shape[2]
    slice = zoom(slice, (1, 256 / x, 256 / y), order=0)
    label = zoom(label, (256 / x, 256 / y), order=0)
    input = torch.from_numpy(slice).unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        out_main = net(input)
        if len(out_main)>1:
            out_main=out_main[0]
        out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        #pred = zoom(out, (x / 256, y / 256), order=0)
        prediction = out
    if np.sum(prediction == 1)==0:
        first_metric = 0,0,0,0
    else:
        first_metric = calculate_metric_percase(prediction == 1, label == 1)
    '''
    img_name = FLAGS.dataset + '_' +case.split('_')[1] + '.png'
    gt_name = FLAGS.dataset + '_' +case.split('_')[1] + '_gt.png'
    pred_name = FLAGS.dataset + '_' +case.split('_')[1] + '_pred.png'
    slice_ = slice.transpose((1, 2, 0)) * 255
    img = Image.fromarray(slice_.astype(np.uint8))
    img.save(test_save_path + img_name)
    label = 1 - label
    label_ = Image.fromarray(label * 255)
    label_.save(test_save_path + gt_name, mode='F')
    prediction_ = prediction.astype(np.uint8) * 255
    prediction = Image.fromarray(prediction_)
    prediction.save(test_save_path + pred_name)
    '''

    '''
    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    '''
    return first_metric

def Inference(FLAGS):
    with open(FLAGS.root_path + '/test_set_3.txt', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])
    snapshot_path = "./model/ACDC_{}_{}_labeled/{}".format(FLAGS.exp, FLAGS.labelnum, FLAGS.model)
    test_save_path = "/home/user/MC-Net-main/predictions/{}_{}_labeled/".format(FLAGS.exp, FLAGS.labelnum)
    
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
    
    args = parser.parse_args()
    net = net_factory(args, net_type = FLAGS.model, in_chns=3, class_num=FLAGS.num_classes)
    #save_model_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    save_model_path = '/home/user/MC-Net-main/model/HAM_HAM10000_DualPerturb_ImgDiff_freqDrop0.5_NoPreRE3H5_60_labeled/self_train/unets_noa_best_model.pth'
    net.load_state_dict(torch.load(save_model_path), strict=True)
    print("init weight from {}".format(save_model_path))
    net.eval()
    first_total = 0.0
    
    for case in tqdm(image_list):
        first_metric = test_single_volume(case, net, test_save_path, FLAGS)
        first_total += np.asarray(first_metric)
    avg_metric = [first_total / len(image_list)]
    return avg_metric, test_save_path

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    metric, test_save_path = Inference(FLAGS)
    print(metric)
    print(metric[0])
    with open(test_save_path+'../performance.txt', 'w') as f:
        f.writelines('metric is {} \n'.format(metric))
        f.writelines('average metric is {}\n'.format(metric[0]))