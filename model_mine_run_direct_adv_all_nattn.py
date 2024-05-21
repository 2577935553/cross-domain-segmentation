import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model_mine_utils import *
from common_utils import *
from torch.utils.data import Dataset,DataLoader
from DenseCRFLoss import *
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
import random
from copy import deepcopy
import copy
from advchain.augmentor import ComposeAdversarialTransformSolver,AdvBias,AdvMorph,AdvNoise,AdvAffine
from itertools import chain



torch.cuda.empty_cache()
import argparse
import time
import logging
import os

seed = 76 # 可以是任何整数 卡住pytorch的随机数
                                
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# def center_crop(images,crop_size,restore=False):
#     if restore:
#         images_restore=np.zeros((images.shape[0],*crop_size)).astype('int8')
#         w,h=images_restore.shape[-2:]
#         crop_w,crop_h=images.shape[-2:]
#         start_w, start_h = w//2-crop_w//2, h//2-crop_h//2
#         end_w, end_h = start_w + crop_w, start_h + crop_h
#         images_restore[..., start_w:end_w, start_h:end_h]=images
#         return images_restore
#     else:
#         crop_w,crop_h=crop_size
#         w,h=images.shape[-2:]
#         start_w, start_h = w//2-crop_w//2, h//2-crop_h//2
#         end_w, end_h = start_w + crop_w, start_h + crop_h
#         cropped_images = images[..., start_w:end_w, start_h:end_h]
#     return cropped_images

# def drop_short_images(images_list,crop_size):
#     images_drop_list=[]
#     for images in images_list:
#         if images.shape[-2:]==tuple(crop_size):
#             images_drop_list.append(images)
#         else:
#             pass
            
#     return images_drop_list



# class CustomDataset(Dataset):
#     def __init__(self, images, labels,use_gpu=True):
#         self.images = images
#         self.labels = labels
#         self.use_gpu=use_gpu
        
#     def __len__(self):
#         return len(self.images)
    
#     def __getitem__(self, index):
#         image = self.images[index]
#         label = self.labels[index]
        
#         # Convert image and label to tensors
#         image_tensor = torch.from_numpy(image).float()
#         label_tensor = torch.from_numpy(label).long()
#         if self.use_gpu:
#             image_tensor=image_tensor.cuda()
#             label_tensor=label_tensor.cuda()
        
#         # # augmentation: 以灰度变换为主：
#         # prob_num=np.random.uniform(0,4)
#         # # 0.25概率线性变换：
#         # if prob_num<1:
#         #     a=np.random.uniform(0.5,2)
#         #     b=np.random.uniform(-1,1)
#         #     image_tensor=a*image_tensor+b
#         #     image_tensor=(image_tensor-image_tensor.min())/(image_tensor.max()-image_tensor.min())
#         # elif prob_num<2:
#         #     image_tensor=


        
#         return image_tensor, label_tensor

# def preprocess_data(imgs):
#     imgs=np.clip(imgs,0,np.percentile(imgs,99))
#     imgs=(imgs-imgs.min())/(imgs.max()-imgs.min()+1e-9)
#     return imgs



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dira-Unet++ Training')
    parser.add_argument('--batch_size', type=int, default=8, 
                            help='input batch size for training ')
    parser.add_argument('--epochs', type=int, default=1000, 
                        help='number of epochs to train ')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='learning rate')
    parser.add_argument('--train_set',type=str,default='test_txt_MMM/test_GE',help='train set path')
    parser.add_argument('--validation_set',type=str,default='test_txt_MMM/MMM_labeled_subj',help='validation set path')
    # parser.add_argument('--train_set',type=str,default='test_txt_MMM/test_GE')
    parser.add_argument('--data_path',type=str,default='/home/fguo/projects/def-gawright/fguo/WHSeg/MMM/Converted/all_labeled/',help='image data path')
    parser.add_argument('--log_path',type=str,default='./log_model_mine_0')
    parser.add_argument('--pred_title',type=str,default='DCM2HCM_attn')
    parser.add_argument('--crop_w',type=int,default=224)
    parser.add_argument('--crop_h',type=int,default=224)
    parser.add_argument('--use_gpu',type=bool,default=True,help='use gpu or not')
    parser.add_argument('--augment',type=bool,default=True)
    parser.add_argument('--attn_features',type=int,default=16)
    parser.add_argument('--loss_edge',type=str,default='dice')



    args=parser.parse_args()
    debug=False
    use_gpu=True
    if use_gpu:
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')

    # now = time.strftime("%Y-%m-%d-%H_%M",time.localtime(time.time()))
    logging.basicConfig(filename=args.log_path+f'/{args.pred_title}.log', level=logging.DEBUG)
    # writer = SummaryWriter(args.log_path+f'/{now}_tensorboard.log')

    train_set=args.train_set

    data_csv_train=pd.read_csv(args.train_set,header=None)
    data_csv_validation=pd.read_csv(args.validation_set,header=None)
    # split train and validation
    objs_train=data_csv_train[0].unique().tolist()
    objs_validation=list(set(data_csv_validation[0].unique().tolist())-set(objs_train))
    objs_train=list(chain.from_iterable([(f"{name}_ED",f"{name}_ES") for name in objs_train]))
    objs_validation=list(chain.from_iterable([(f"{name}_ED",f"{name}_ES") for name in objs_validation]))
    crop_size=(args.crop_w,args.crop_h)
    
    # load Data
    imgs_train=np.concatenate(drop_short_images([center_crop(preprocess_data(nib.load(args.data_path+names+'/Img_resamp.nii.gz').get_fdata().swapaxes(1,2).swapaxes(0,1)),crop_size) for names in objs_train],crop_size),axis=0)
    imgs_validation_list=drop_short_images([center_crop(preprocess_data(nib.load(args.data_path+names+'/Img_resamp.nii.gz').get_fdata().swapaxes(1,2).swapaxes(0,1)),crop_size) for names in objs_validation],crop_size)
    imgs_validation=np.concatenate(imgs_validation_list,axis=0)
    labels_train=np.concatenate(drop_short_images([center_crop(nib.load(args.data_path+names+'/GT_resamp.nii.gz').get_fdata().swapaxes(1,2).swapaxes(0,1),crop_size) for names in objs_train],crop_size),axis=0)
    labels_validation_list=drop_short_images([center_crop(nib.load(args.data_path+names+'/GT_resamp.nii.gz').get_fdata().swapaxes(1,2).swapaxes(0,1),crop_size) for names in objs_validation],crop_size)
    labels_validation=np.concatenate(labels_validation_list,axis=0)

    imgs_validation_test_list=drop_short_images_test([(center_crop(preprocess_data(nib.load(args.data_path+names+'/Img_resamp.nii.gz').get_fdata().swapaxes(1,2).swapaxes(0,1)),crop_size),names) for names in objs_validation],crop_size)
    labels_validation_test_list=drop_short_images_test([(center_crop(preprocess_data(nib.load(args.data_path+names+'/Img_resamp.nii.gz').get_fdata().swapaxes(1,2).swapaxes(0,1)),crop_size),names) for names in objs_validation],crop_size)


    imgs_train=imgs_train[:,None,...]
    # 对数据预处理：
    

    if args.augment:
        dataset_train=Augmentation_Dataset(imgs_train,labels_train,use_gpu=True)
    else:
        dataset_train=CustomDataset(imgs_train,labels_train,use_gpu=True)
    dataloader_train=DataLoader(dataset_train,batch_size=args.batch_size,drop_last=True,shuffle=True)
    gl_loss=GradientLoss(channel_mean=False,loss=args.loss_edge)
    # augment model:
    augmentor_bias= AdvBias(
                    config_dict={'epsilon':0.3,
                    'control_point_spacing':[crop_size[0]//2,crop_size[1]//2],
                    'downscale':4,
                    'data_size':(args.batch_size,1,crop_size[0],crop_size[1]),
                    'interpolation_order':3,
                    'init_mode':'random',
                    'space':'log'},
                    debug=debug,use_gpu=use_gpu)

    augmentor_noise= AdvNoise( config_dict={'epsilon':1,
                'xi':1e-6,
                    'data_size':(args.batch_size,1,crop_size[0],crop_size[1])},
                    debug=debug,use_gpu=use_gpu)

    augmentor_affine= AdvAffine( config_dict={
                    'rot':30.0/180,
                    'scale_x':0.2,
                    'scale_y':0.2,
                    'shift_x':0.1,
                    'shift_y':0.1,
                    'data_size':(args.batch_size,1,crop_size[0],crop_size[1]),
                    'forward_interp':'bilinear',
                    'backward_interp':'bilinear'},
                    image_padding_mode="zeros", ## change it other int/float values for padding or other padding mode like "reflection", "border" or "zeros," "zeros" is the default value
                    debug=debug,use_gpu=use_gpu)
    augmentor_morph= AdvMorph(
                config_dict=
                {'epsilon':1.5,
                    'data_size':(args.batch_size,1,crop_size[0],crop_size[1]),
                    'vector_size':[crop_size[0]//16,crop_size[1]//16],
                    'forward_interp':'bilinear',
                    'backward_interp':'bilinear'
                    }, 
                    image_padding_mode="zeros", ## change it other int/float values for padding or other padding mode like "reflection", "border" or "zeros," "zeros" is the default value
                    debug=debug,use_gpu=use_gpu)
    transform_chain=[augmentor_noise,augmentor_bias,augmentor_morph,augmentor_affine] 
    solver = ComposeAdversarialTransformSolver(
        chain_of_transforms=transform_chain,
        divergence_types = ['mse','contour','kl'], ### you can also change it to 'kl'
        divergence_weights=[1.0,1.0,1.0],
        use_gpu= use_gpu,
        debug=debug, ## turn off debugging information when training your model
        if_norm_image=False,
        )
    solver.init_random_transformation()
        


    # model
    encoder_name_ls=['efficientnet-b2','dpn68','densenet121','timm-regnety_016']

    model=Main_Model(image_size=crop_size[0],num_models=4,with_edge=True,with_attn=False,encoder_name_ls=encoder_name_ls,attn_features=args.attn_features).cuda()
    # model=smp.UnetPlusPlus(encoder_name='densenet121',in_channels=1,classes=4).cuda()
    optimizer = Ranger(model.parameters(), lr=args.lr)
    diceloss_4_fn=smp.losses.DiceLoss(mode='multiclass',classes=[0,1,2,3]).cuda()
    diceloss_3_fn=smp.losses.DiceLoss(mode='multiclass',classes=[1,2,3]).cuda()
    celoss_fn=nn.CrossEntropyLoss().cuda() # pred,one-hot label
    crfloss = DenseCRFLoss(1e-9,15,100,0.5).cuda()
    mseloss=nn.MSELoss()

    best_dice=0
    for epoch in tqdm(range(args.epochs)):
        model.train()
        loss_n_class_ls=[]
        loss_class_consistency_ls=[]
        loss_edge_ls=[]
        # loss_dice_crf_ls=[]
        loss_dice_normal_ls=[]
        loss_sum_ls=[]
        reg_loss_ls=[]
        for img,label in tqdm(dataloader_train):
            # print(img.shape,label.shape)
            
            # pred_x_ls,pred_dice=model(img)
            # hot_label=F.one_hot(label.long().squeeze(),num_classes=4).permute(0,3,1,2)
            reg_loss=solver.adversarial_training(
                    data = img,model=model,
                    n_iter = 1, ## set up  the number of iterations for updating the transformation model.
                    lazy_load = [False]*len(transform_chain), 
                    optimize_flags = [True]*len(transform_chain),  ## you can also turn off adversarial training for one particular transformation
                    step_sizes = 1) ## set up step size, you can also change it to a list of step sizes, so that different transformation have different step size
            solver.reset_transformation()
            model.train()
            pred_x_ls,pred_edge,pred_dice=model(img)
            loss_n_class=0.2*celoss_fn(torch.cat(pred_x_ls,dim=1),label.long())+0.8*diceloss_4_fn(torch.cat(pred_x_ls,dim=1),label)

            loss_class_consistency=mseloss(torch.cat(pred_x_ls,dim=1),pred_dice)
            loss_edge=gl_loss(pred_edge,label.squeeze())

            # loss_dice_normal=max(1-0.01*epoch,0.2)*celoss_fn(pred_dice,label.long())+min(0.01*epoch,0.8)*diceloss_3_fn(pred_dice,label)
            loss_dice_normal=0.2*celoss_fn(pred_dice,label.long())+0.8*diceloss_3_fn(pred_dice,label)

            loss_sum=loss_n_class+loss_class_consistency+loss_dice_normal+loss_edge+reg_loss*3
            # loss_sum=loss_n_class+loss_class_consistency+loss_dice_normal

            loss_n_class_ls.append(loss_n_class.item())
            loss_class_consistency_ls.append(loss_class_consistency.item())
            loss_edge_ls.append(loss_edge.item())
            loss_dice_normal_ls.append(loss_dice_normal.item())
            loss_sum_ls.append(loss_sum.item())
            reg_loss_ls.append((reg_loss*3).item())


            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()
        # if epoch>20:
        logging.info(f'epoch:{epoch},loss_n_class:{np.mean(loss_n_class_ls)},loss_class_consistency:{np.mean(loss_class_consistency_ls)},loss_edge:{np.mean(loss_edge_ls)},loss_dice_normal:{np.mean(loss_dice_normal_ls)},reg_loss:{np.mean(reg_loss_ls)},loss_sum:{np.mean(loss_sum_ls)}')
        # logging.info(f'epoch:{epoch},loss_n_class:{np.mean(loss_n_class_ls)},loss_class_consistency:{np.mean(loss_class_consistency_ls)},loss_dice_normal:{np.mean(loss_dice_normal_ls)},loss_sum:{np.mean(loss_sum_ls)}')

        model.eval()
        with torch.no_grad():
            dsc_val_ls=[]
            for img_,label_ in zip(imgs_validation_list,labels_validation_list):
                img_=torch.from_numpy(img_[:,np.newaxis,...]).cuda().float()
                label_=torch.from_numpy(label_).cuda()
                dsc_val=[]
                _,_,output_val=model(img_)
                _,pred = torch.max(output_val,1)
                for i in range(3):
                    dsc_val.append(2*(((pred==(i+1))*(label_==(i+1))).sum()/((pred==(i+1)).sum()+(label_==(i+1)).sum())).item())
                dsc_val_ls.append(dsc_val)
            dsc_val_1,dsc_val_2,dsc_val_3=[np.mean([dsc_val_ls[i][j] for i in range(len(dsc_val_ls))]) for j in range(3)]
            dsc_val=np.mean(dsc_val_ls)
            if dsc_val>best_dice:
                model_test=copy.deepcopy(model)
                best_dice=dsc_val
                torch.save(model.state_dict(),args.log_path+f'/{args.pred_title}_best_model.pth')
            logging.info(f'epoch:{epoch},dsc_val:{dsc_val},dsc_val_1:{dsc_val_1},dsc_val_2:{dsc_val_2},dsc_val_3:{dsc_val_3}')
    test_output_dir=f'{args.log_path}/{args.pred_title}' # 每一个log对应一个文件和文件夹
    generate_pred(model_test,imgs_validation_test_list,args.data_path,test_output_dir,device=device,title=args.pred_title)





