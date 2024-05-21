import os

import nibabel as nib
import numpy as np
import pandas as pd

import image_utils
import argparse
from itertools import chain
# from image_utils import np_categorical_dice_3d, np_categorical_jaccard_3d, np_categorical_assd_hd
parser = argparse.ArgumentParser(description='dira-Unet++ Training')

# parser.add_argument('--train_set',type=str,default=None)
parser.add_argument('--validation_set',type=str,default='/home/fguo/projects/def-gawright/fguo/GuoLab_students/szhang/advchain-master/center2_Philips_HCM.txt',help='validation set path')
# parser.add_argument('--part_val',type=bool,default=False)
parser.add_argument('--data_path',type=str,default='/home/fguo/projects/def-gawright/fguo/WHSeg/MMM/Converted/all_labeled/',help='image data path')
parser.add_argument('--log_path',type=str,default='./log_model_mine_0')
parser.add_argument('--use_gpu',type=bool,default=True,help='use gpu or not')
# parser.add_argument('--test_output_dir',type=str,default='/home/fguo/projects/def-gawright/fguo/GuoLab_students/szhang/advchain-master/log_model_mine_0/DCM2HCM_attn',help='test_model.pth')
parser.add_argument('--pred_title',type=str,default='DCM2HCM_attn')
parser.add_argument('--name_csv',type=str,default='DCM2HCM_attn')

args=parser.parse_args()

data_directory = f'{args.log_path}/{args.pred_title}'
print(data_directory)
num_classes = 4

name_list = pd.read_csv(args.validation_set, header=None)[0].tolist()
name_list=list(chain.from_iterable([(f"{name}_ED",f"{name}_ES") for name in name_list]))

gt_label_list = ['/home/fguo/projects/def-gawright/fguo/WHSeg/MMM/Converted/all_labeled/{}/GT_resamp.nii.gz'.format(i) for i in name_list]
pred_list = ['{}/Pred_{}.nii.gz'.format(i,args.pred_title) for i in name_list]
output_file = f'{args.name_csv}.csv'
# pred_list = ['{}/PredAvgCMF.nii.gz'.format(i,i) for i in name_list]
# output_file = 'PredAvgCMF_Res.csv'
# pred_list = ['{}/PredAvgCMF_orig_space.nii.gz'.format(i,i) for i in name_list]
# output_file = 'PredAvgCMF_orig_Res.csv'
# pred_list = ['{}/MLKC_out.nii.gz'.format(i,i) for i in name_list]
# output_file = 'MLKC_Res.csv'
# pred_list = ['{}/MLKC_4DAvg.nii.gz'.format(i,i) for i in name_list]
# output_file = 'MLKC_4DAvg_Res.csv'
# pred_list = ['{}/MLKC_4Dout.nii.gz'.format(i,i) for i in name_list]
# output_file = 'MLKC_4D_Res.csv'
# pred_list = ['{}/MLKC_4D_4DAvg.nii.gz'.format(i,i) for i in name_list]
# output_file = 'MLKC_4D_4DAvg_Res.csv'
# pred_list = ['{}/MLKC_4D_4Dout.nii.gz'.format(i,i) for i in name_list]
# output_file = 'MLKC_4D_4D_Res.csv'
# pred_list = ['{}/MLKC_4D_4D_2Dout.nii.gz'.format(i,i) for i in name_list]
# output_file = 'MLKC_4D_4D_2D_Res.csv'
# pred_list = ['{}/MLKC_4D_4D_2DoutAvg.nii.gz'.format(i,i) for i in name_list]
# output_file = 'MLKC_4D_4D_2DAvg_Res.csv'


dice_classes = ['DiceClass{}'.format(i) for i in range(1, num_classes)]
assd_np_classes = ['AssdNpClass{}'.format(i) for i in range(1, num_classes)]
hd_np_classes = ['HdNpClass{}'.format(i) for i in range(1, num_classes)]
# jaccard_classes = ['JaccardClass{}'.format(i) for i in range(1, num_classes)]
volE_classes = ['VolEClass{}'.format(i) for i in range(1, num_classes)]
volP_classes = ['VolPClass{}'.format(i) for i in range(1, num_classes)]
volPred_classes = ['VolPredClass{}'.format(i) for i in range(1, num_classes)]
volGT_classes = ['VolGTClass{}'.format(i) for i in range(1, num_classes)]

df = pd.DataFrame(columns=['Subject'] + dice_classes + assd_np_classes + hd_np_classes + volE_classes + volP_classes + volPred_classes + volGT_classes)

for gt_label_filename, pred_filename in zip(gt_label_list, pred_list):
  
  try:
    print('Processing GT: {}, Prediction: {}'.format(gt_label_filename, pred_filename))

    label_nii = nib.load(os.path.join(data_directory, gt_label_filename))
    pixel_spacing = label_nii.header['pixdim'][1:4]
    print(pixel_spacing)
    label = label_nii.get_data().astype(np.int8).squeeze()
    
    print(os.path.join(data_directory, pred_filename))
    pred_nii = nib.load(os.path.join(data_directory, pred_filename))
    pred = pred_nii.get_data().astype(np.int8).squeeze()
    print('get')
    # pred = pred_nii.get_data()[:, :, :, 1]

    dice = image_utils.np_categorical_dice_3d(pred, label, num_classes)
    # jaccard = image_utils.np_categorical_jaccard_3d(pred, label, num_classes)
    # print(pred.shape,label.shape)
    assd_np, hd_np = image_utils.np_categorical_assd_hd(pred, label, num_classes, pixel_spacing[0:2])
    # assd_np, hd_np = image_utils.np_categorical_assd_hd_3d(pred, label, num_classes, pixel_spacing[0:3])
    volE_np, volP_np, vol_pred, vol_gt= image_utils.np_categorical_volume_3d(pred, label, num_classes, pixel_spacing[0:3])

    subject_slice_df = {}
    subject_slice_df['Subject'] = gt_label_filename

    for j in range(1, num_classes):
      subject_slice_df['DiceClass{}'.format(j)] = dice[j]
      subject_slice_df['AssdNpClass{}'.format(j)] = assd_np[j]
      subject_slice_df['HdNpClass{}'.format(j)] = hd_np[j]
      subject_slice_df['VolEClass{}'.format(j)] = volE_np[j]
      subject_slice_df['VolPClass{}'.format(j)] = volP_np[j]
      subject_slice_df['VolPredClass{}'.format(j)] = vol_pred[j]
      subject_slice_df['VolGTClass{}'.format(j)] = vol_gt[j]

    df = df.append(subject_slice_df, ignore_index=True)
  except:
    pass
  
average = df.mean(axis=0, skipna=True)
average['Subject'] = 'Mean'

stddev = df.std(axis=0, ddof=0, skipna=True)
stddev['Subject'] = 'SD'

max = df.max(axis=0, skipna=True)
max['Subject'] = 'Max'

min = df.min(axis=0, skipna=True)
min['Subject'] = 'Min'

df = df.append([average, stddev, max, min])
df = df.round(3)
df.to_csv(os.path.join(data_directory, output_file), index=False, na_rep='nan')
