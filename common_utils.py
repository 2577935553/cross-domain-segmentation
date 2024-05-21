import statistics
import shutil
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# from model_mine_utils import *
from torch.utils.data import Dataset,DataLoader
import math
import seaborn as sns
import torch
import torchvision
import sys
import pandas as pd
import nibabel as nib
from scipy import ndimage
from numpy.linalg import inv
from scipy.ndimage.filters import gaussian_filter
import csv
from abc import ABC
from stretching import elastic_transform_by_scale
import cv2
## train
from tqdm import tqdm



torch.cuda.empty_cache()
import argparse
import time
import logging
import os
import image_utils

def drop_short_images_test(images_list,crop_size):
    images_drop_list=[]
    for images,name in images_list:
        if images.shape[-2:]==tuple(crop_size):
            images_drop_list.append((images,name))
        else:
            pass
            
    return images_drop_list
def drop_short_images(images_list,crop_size):
    images_drop_list=[]
    for images in images_list:
        if images.shape[-2:]==tuple(crop_size):
            images_drop_list.append(images)
        else:
            pass
            
    return images_drop_list
def center_crop(images,crop_size,restore=False):
    if restore:
        images_restore=np.zeros((images.shape[0],*crop_size)).astype('int8')
        w,h=images_restore.shape[-2:]
        crop_w,crop_h=images.shape[-2:]
        start_w, start_h = w//2-crop_w//2, h//2-crop_h//2
        end_w, end_h = start_w + crop_w, start_h + crop_h
        images_restore[..., start_w:end_w, start_h:end_h]=images
        return images_restore
    else:
        crop_w,crop_h=crop_size
        w,h=images.shape[-2:]
        start_w, start_h = w//2-crop_w//2, h//2-crop_h//2
        end_w, end_h = start_w + crop_w, start_h + crop_h
        cropped_images = images[..., start_w:end_w, start_h:end_h]
    return cropped_images

def preprocess_data(imgs):
    imgs=np.clip(imgs,0,np.percentile(imgs,99))
    imgs=(imgs-imgs.min())/(imgs.max()-imgs.min()+1e-9)
    return imgs

def generate_pred(model_test,imgs_validation_list,data_path,test_output_dir,device='cuda',title="DCM2HCM",output_num=3):
    """
    params:
    model_test:预测模型
    imgs_validation_list:[(name:str,imgs:np.array)]
    data_path:原数据的根路径
    test_output_dir:预测路径
    """
    num_classes = 4
    name_list=[imgs_validation_list[idx][1] for idx in range(len(imgs_validation_list))]
    # name_list = pd.read_csv('center2_Philips_HCM.txt', header=None)[0].tolist()

    gt_label_list = ['/home/fguo/projects/def-gawright/fguo/WHSeg/MMM/Converted/all_labeled/{}/GT_resamp.nii.gz'.format(i,i) for i in name_list]
    pred_list = [f'{test_output_dir}/{name}/Pred_{title}.nii.gz' for name in name_list]
    output_file=f'{test_output_dir}/Pred_{title}.csv'
    if not os.path.exists(f'{test_output_dir}'):
        os.mkdir(f'{test_output_dir}')
    with torch.no_grad():
        for idx in range(len(imgs_validation_list)):
            name=imgs_validation_list[idx][1]
            imgs_=torch.from_numpy(imgs_validation_list[idx][0][:,np.newaxis,...]).to(device).float()
            if output_num==3:
                _,_,output=model_test(imgs_)
            elif output_num==2:
                _,output=model_test(imgs_)
            else:
                output=model_test(imgs_)

            _,pred=torch.max(output,1)
            pred_np=pred.squeeze().detach().cpu().numpy().astype('int8')
            # pred_np=labels_validation_list[idx]

            
            dir_path=f'{test_output_dir}/{name}'
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            source_files = [data_path+name+'/Img_resamp.nii.gz',data_path+name+'/GT_resamp.nii.gz']
            destination_files = [dir_path+'/Img_resamp.nii.gz',dir_path+'/GT_resamp.nii.gz']
            nib_gt = nib.load(source_files[1])
            gt = nib_gt.get_fdata().astype('int8')
            pred_np=(center_crop(pred_np,gt.shape[:2],restore=True)).swapaxes(0,1).swapaxes(1,2)
            nib_pred = nib.Nifti1Image(pred_np, None, header = nib_gt.header)
            
            [shutil.copy(source_files[i], destination_files[i]) for i in range(2)]
            nib.save(nib_pred,dir_path+f'/Pred_{title}.nii.gz')

# def calculate_metrics_bk()

class CustomDataset(Dataset):
    def __init__(self, images, labels,use_gpu=True):
        self.images = images
        self.labels = labels
        self.use_gpu=use_gpu
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        
        # Convert image and label to tensors
        image_tensor = torch.from_numpy(image).float()
        label_tensor = torch.from_numpy(label).long()
        if self.use_gpu:
            image_tensor=image_tensor.cuda()
            label_tensor=label_tensor.cuda()
            
        return image_tensor, label_tensor


def augment_data(image, label, shift=30, rotate=60, scale=0.2, intensity=0.2, flip=False):
    aug_image = np.zeros_like(image)
    aug_label = np.zeros_like(label)
    # for i in range(image.shape[0]):
    tem_image = image
    tem_label = label

    # For each image slice, generate random affine transformation parameters
    # using the Gaussian distribution
    shift_val = [int(shift * np.random.uniform(-1, 1)),
                    int(shift * np.random.uniform(-1, 1))]
    rotate_val = rotate * np.random.uniform(-1, 1)
    scale_val = 1 + scale * np.random.uniform(-1, 1)

    tem_image = tem_image * (1 + intensity * np.random.uniform(-1, 1))

    # Apply the affine transformation (rotation + scale + shift) to the image
    row, col = tem_image.shape
    M = cv2.getRotationMatrix2D((row / 2, col / 2), rotate_val, 1.0 / scale_val)
    M[:, 2] += shift_val
    M = np.concatenate((M, np.array([[0, 0, 1]])), axis=0)

    aug_image = ndimage.interpolation.affine_transform(tem_image, inv(M), order=1)
    aug_label = ndimage.interpolation.affine_transform(tem_label, inv(M), order=0)

    return aug_image, aug_label

def stack_augmentation(image, labels, stack_num=1):
    """

    :param image:->[b,h,w] hw
    :param labels:-> [b,h,w] hw
    :param stack_num: number of stacks
    :return: ->[b,h,w] hw
    """
    
    img = image.copy()
    label = labels.copy()

    for i in range(stack_num):
        prob = np.random.uniform(0,1)

        if prob >0.5: # 50%的数据做数据增强
            img, label = augment_data(img, label, shift=0,
                                                    rotate=30, scale=0.2, intensity=0.2, flip=False)

            shift_max = 10
            shift = int(shift_max * np.random.uniform(-1, 1))

        prob = np.random.uniform(0,1)

        if prob >0.5: 
            img = gamma_correction(img)

        prob = np.random.randint(0, 2, [1]) # noise
        if prob == 0:
            std = np.random.uniform(0.05, 0.1)
            img = img + np.random.normal(0, scale=std, size=img.shape)

        prob = np.random.randint(0, 2, [1])
        if prob == 0:
            # for j in range(img.shape[0]):
            std = np.random.uniform(0.25, 1.5)

            img = gaussian_filter(img, std, order=0)
        
        
        # prob = np.random.randint(0, 2, [1]) # gaussian filter
        # if prob == 0:
        #     alpha = np.random.uniform(5, 15)
        #     std = np.random.uniform(0.25, 1.5)
    
        #     i_blurred = gaussian_filter(img, std, order=0)
        #     i_filterblurred = gaussian_filter(i_blurred, std, order=0)
        #     img = i_blurred + alpha * (i_blurred - i_filterblurred)

        # prob = np.random.randint(0, 2, [1])
        # if prob == 0:
        #     I = img.copy()
        #     inputs = torch.rand(1, 1, 9, 9)
        #     grid_x = torch.linspace(-1, 1, img.shape[0]).view(1, -1, 1, 1).repeat(1, 1, img.shape[0], 1)
        #     grid_y = torch.linspace(-1, 1, img.shape[0]).view(1, 1, -1, 1).repeat(1, img.shape[0], 1, 1)
        #     grid = torch.cat([grid_x, grid_y], dim=3)
        #     outs = F.grid_sample(inputs, grid, mode='bilinear')
        #     outs = outs.numpy().squeeze()
        #     img = outs * I

        
        prob = np.random.uniform(0,1)
        if prob >0.2: # 80%的数据做数据增强
            Ig = img.copy()
            Lb = label.copy()
            
            prob=np.random.uniform(0,2)
            if prob<1: #缩小:扩张1：1
                deform_x = np.random.uniform(0.5, 1)
                deform_y = np.random.uniform(0.5, 1)
            else:
                deform_x=np.random.uniform(1,10)
                deform_y=np.random.uniform(1,10)
            # label_range = np.argwhere(label[s, :, :])
            x1,y1=np.where(label==1)
            x2,y2=np.where(label==2)
            x3,y3=np.where(label==3)
            x=np.concatenate((x1,x2,x3))
            y=np.concatenate((y1,y2,y3))
            if x.size==0:
                width_x=int(np.random.uniform(20,40))
                width_y=int(np.random.uniform(20,40))
                smooth_x=np.random.uniform(10,30)
                smooth_y=np.random.uniform(10,30)
                x_mid,y_mid=img.shape[0]//2,img.shape[1]//2
            # if label_range.size == 0:
            #     continue
            else:
                x_min,x_max=np.min(x),np.max(x)
                y_min,y_max=np.min(y),np.max(y)
                x_mid,y_mid=(x_min+x_max)//2,(y_min+y_max)//2
                width_x=int((x_max-x_min-1)//2*np.random.uniform(0.8,1.0))
                width_y=int((y_max-y_min-1)//2*np.random.uniform(0.8,1.0))
                smooth_x=np.random.uniform(10,width_x)*np.random.uniform(0.8,1.2)
                smooth_y=np.random.uniform(10,width_y)*np.random.uniform(0.8,1.2)

            # (x_start, y_start), (x_stop, y_stop) = label_range.min(0), label_range.max(0) + 1
            # x_mid = (x_start + x_stop) // 2
            # y_mid = (y_start + y_stop) // 2
            img, label = elastic_transform_by_scale(Ig, Lb,
                                                        scale=(deform_x, deform_y),
                                                        centre=(x_mid, y_mid),
                                                        width=(width_x, width_y),
                                                        smooth=(smooth_x,smooth_y))
        prob=np.random.uniform(0,1)
        if prob>0.1:
            # 对图像做sin风格变换
            m=np.random.random()*8*np.pi/2
            n=np.random.random()*2*np.pi
            img=np.sin(m*img+n)
            img=(img-img.min())/(img.max()-img.min()) #norm
    return img, label

def gamma_correction(img):
    # img : b,h,w
    Image = img.copy()
    # for i in range(img.shape[0]):
    gamma = np.random.uniform(0.5, 1.5, [1])
    Image = np.sign(img) * (np.abs(img)) ** gamma
    # 对每个像素做gamma变换
    return Image
   







class Augmentation_Dataset(Dataset):
    def __init__(self, imgs, labels, image_size=(224, 224), mode='train', use_gpu=True):
        # super().__init__(imgs, labels, image_size, mode, use_gpu)

        
        self.images = imgs
        self.labels = labels
        self.use_gpu=use_gpu

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image=self.images[idx].squeeze() # h*w
        label=self.labels[idx].squeeze()
        # augmentation:
        image_aug,label_aug=stack_augmentation(image,label)
        image_tensor = torch.from_numpy(image_aug).float()
        label_tensor = torch.from_numpy(label_aug).long()
        if self.use_gpu:
            image_tensor=image_tensor.cuda()
            label_tensor=label_tensor.cuda()



        return image_tensor.unsqueeze(0), label_tensor
