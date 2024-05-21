import statistics
import shutil
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model_mine_utils import *
from torch.utils.data import Dataset,DataLoader
from common_utils import *
import math
import seaborn as sns
import torch
import torchvision
import sys
import pandas as pd
import nibabel as nib
## train
from tqdm import tqdm
from torch_optimizer import Ranger
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

torch.cuda.empty_cache()
import argparse
import time
import logging
import os
from itertools import  chain




if __name__=="__main__":
    parser = argparse.ArgumentParser(description='argparse init')
    parser.add_argument('--crop_w',type=int,default=192,help='crop size w')
    parser.add_argument('--crop_h',type=int,default=192,help='crop size h')
    parser.add_argument('--validation_set',type=str,default='/home/fguo/projects/def-gawright/fguo/GuoLab_students/szhang/advchain-master/test_txt_MMM/MMM_labeled_subj',help='validation set path')
    parser.add_argument('--data_path',type=str,default='/home/fguo/projects/def-gawright/fguo/WHSeg/MMM/Converted/all_labeled/',help='image data path')
    parser.add_argument('--test_model',type=str,default='/home/fguo/projects/def-gawright/fguo/GuoLab_students/szhang/advchain-master/log_model_mine_all_nattn/train_on_Philips_best_model.pth',help='test_model.pth')
    parser.add_argument('--test_output_dir',type=str,default='/home/fguo/projects/def-gawright/fguo/GuoLab_students/szhang/advchain-master/log_model_mine_all_nattn/train_on_Philips',help='test_model.pth')
    parser.add_argument('--pred_title',type=str,default='train_on_Philips')

    args=parser.parse_args()

    batch_size = 1
    if torch.cuda.is_available():
        device='cuda'
    else:
        device='cpu'


    crop_size=[args.crop_w,args.crop_h]
    data_path=args.data_path
    test_model=args.test_model
    validation_set=args.validation_set
    test_output_dir=args.test_output_dir

    data_csv_validation=pd.read_csv(validation_set,header=None)
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)

    objs_validation=data_csv_validation[0].unique().tolist()
    objs_validation=list(chain.from_iterable([(f'{name}_ED',f'{name}_ES') for name in objs_validation]))


    imgs_validation_list=drop_short_images_test([(center_crop(preprocess_data(nib.load(args.data_path+names+'/Img_resamp.nii.gz').get_fdata().swapaxes(1,2).swapaxes(0,1)),crop_size),names) for names in objs_validation],crop_size)
    labels_validation_list=drop_short_images_test([(center_crop(preprocess_data(nib.load(args.data_path+names+'/Img_resamp.nii.gz').get_fdata().swapaxes(1,2).swapaxes(0,1)),crop_size),names) for names in objs_validation],crop_size)
    
    
    # model_test=UnetPP(encoder_name='densenet121',
    #     encoder_weights='imagenet',
    #     encoder_depth=5,
    #     decoder_channels=[256, 128, 64, 32, 16],
    #     in_channels=1,
    #     classes=4
    #     )
    encoder_name_ls=['efficientnet-b2','dpn68','densenet121','timm-regnety_016']

    model_test=Main_Model(image_size=crop_size[0],num_models=4,with_edge=True,with_attn=False,encoder_name_ls=encoder_name_ls,attn_features=16).cuda()

    model_test.load_state_dict(torch.load(test_model,map_location=torch.device(device)))
    model_test=model_test.to(device)
    model_test.eval()
    generate_pred(model_test,imgs_validation_list,data_path,test_output_dir,device='cuda',title=args.pred_title)


    # with torch.no_grad():
    #     for idx in range(len(imgs_validation_list)):
    #         name=imgs_validation_list[idx][1]
    #         imgs_=torch.from_numpy(imgs_validation_list[idx][0][:,np.newaxis,...]).to(device).float()
    #         _,_,output=model_test(imgs_)
    #         _,pred=torch.max(output,1)
    #         pred_np=pred.squeeze().detach().cpu().numpy().astype('int8')
    #         # pred_np=labels_validation_list[idx]

            
    #         dir_path=f'{test_output_dir}/{name}'
    #         if not os.path.exists(dir_path):
    #             os.mkdir(dir_path)
    #         source_files = [data_path+name+'/Img_resamp.nii.gz',data_path+name+'/GT_resamp.nii.gz']
    #         destination_files = [dir_path+'/Img_resamp.nii.gz',dir_path+'/GT_resamp.nii.gz']
    #         nib_gt = nib.load(source_files[1])
    #         gt = nib_gt.get_fdata().astype('int8')
    #         pred_np=(center_crop(pred_np,gt.shape[:2],restore=True)).swapaxes(0,1).swapaxes(1,2)
    #         nib_pred = nib.Nifti1Image(pred_np, None, header = nib_gt.header)
            
    #         [shutil.copy(source_files[i], destination_files[i]) for i in range(2)]
    #         nib.save(nib_pred,dir_path+'/Pred.nii.gz')
            

    
    

    