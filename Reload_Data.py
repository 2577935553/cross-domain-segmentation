import csv
from abc import ABC
from stretching import elastic_transform_by_scale
import cv2
import numpy as np
import torch
from torch.utils import data
import nibabel as nib
from scipy import ndimage
from numpy.linalg import inv
from scipy.ndimage.filters import gaussian_filter
import torch.nn.functional as F
from torch.utils.data import Dataset



def augment_data(self, image, label, shift=30, rotate=60, scale=0.2, intensity=0.2, flip=False):
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
            img, label = augment_data_batch(img, label, shift=0,
                                                    rotate=30, scale=0.2, intensity=0.2, flip=False)

            shift_max = 10
            shift = int(shift_max * np.random.uniform(-1, 1))

        prob = np.random.uniform(0,1)

        if prob >0.5: 
            img = gamma_correction(img)

        prob = np.random.randint(0, 2, [1]) # noise
        if prob == 0:
            std = np.random.uniform(0.05, 0.1)
            img = img + np.random.normal(0, scale=std, size=img[j].shape)
        
        
        prob = np.random.randint(0, 2, [1]) # gaussian filter
        if prob == 0:
            alpha = np.random.uniform(5, 15)
            std = np.random.uniform(0.25, 1.5)
    
            i_blurred = gaussian_filter(img, std, order=0)
            i_filterblurred = gaussian_filter(i_blurred, std, order=0)
            img = i_blurred + alpha * (i_blurred - i_filterblurred)

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
                x_mid,y_mid=img.shape[1]//2,img.shape[2]//2
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
        super().__init__(imgs, labels, image_size, mode, use_gpu)

        
        self.images = images
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



        return image_tensor.unsqueeze(1), label_tensor
