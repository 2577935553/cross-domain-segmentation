import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from pytorch_ssim import SSIM
from scipy.ndimage.morphology import distance_transform_edt
import cv2

class Segmentation_part(nn.Module):
    def __init__(self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,):

        super().__init__()
        self.model=smp.UnetPlusPlus(encoder_name=encoder_name,in_channels=1,classes=4)

    
    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.model.encoder(x)
        decoder_output = self.model.decoder(*features)

        return decoder_output
    
class CrossAttention(nn.Module):
    def __init__(self, dim,in_channels, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.conv_cat=nn.Conv2d(in_channels*2,in_channels,kernel_size=3,padding=3//2)

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x1,x2):

        B, N, H, W = x1.shape
        C=H*W
        # print(x1.shape)
        x_cat=torch.cat([x1,x2],dim=1)
        x_cat=self.conv_cat(x_cat).reshape(B,N,C)


        q = self.wq(x1.reshape(B,N,C)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  
        k = self.wk(x_cat).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x_cat).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BHN(C/H) @ BH(C/H)N -> BHNN
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)   # (BHNN @ BHN(C/H)) -> BHN(C/H) -> BNH(C/H) -> BNC
        x_attn = self.proj(x_attn).reshape(B,N,H,W)+x1


        # x = self.proj_drop(x)
        return x_attn

class Get_feature(nn.Sequential):
    def __init__(self,in_channels,attn_features):
        pool=nn.AvgPool2d(4)
        conv1=nn.Conv2d(in_channels,attn_features,kernel_size=3,padding=3//2)
        # conv2=nn.Conv2d(in_channels*2,in_channels*4,kernel_size=3,padding='same')
        # conv3=nn.Conv2d(in_channels*4,in_channels*8,kernel_size=3,padding='same')
        # activation=None
        super().__init__(pool,conv1)

class Segmentation_Block(nn.Module):
    def __init__(self,encoder_name="densenet121",attn_features=32):
        super().__init__()
        self.model_c=Segmentation_part(encoder_name="densenet121")
        self.get_fc=Get_feature(16,attn_features) # return 16*2
        self.seg_c=nn.Conv2d(16,1,kernel_size=3,padding=3//2)
        
    def forward(self,x):
        x=self.model_c(x)
        f_x=self.get_fc(x)
        pred_x=self.seg_c(x)

        return x,pred_x,f_x

class CrossAttention_Map(nn.Module):
    def __init__(self,image_size,attn_features,numbers) -> None:
        super().__init__()
        self.numbers=numbers

        self.attn_map=nn.ModuleDict({f"{i}_{j}":CrossAttention(image_size//4*image_size//4,attn_features).cuda() for i in range(numbers) for j in range(numbers)})
        self.conv_re_ls=nn.ModuleList([nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=4),nn.Conv2d(attn_features*numbers,16,kernel_size=3,padding=3//2)).cuda() for _ in range(numbers)])
        self.conv_cat=nn.Conv2d(16*numbers,16*2,kernel_size=3,padding=3//2)

    
    def forward(self,x_ls,f_x_ls):
        attn_x=torch.cat([self.conv_re_ls[i](torch.cat([(self.attn_map[f"{i}_{j}"](f_x_ls[i],f_x_ls[j])) for j in range(self.numbers)],dim=1))+x_ls[i] for i in range(self.numbers)],dim=1)
        attn_x=self.conv_cat(attn_x)

        return attn_x  #channels=32

class Conv_edge(nn.Sequential):
    def __init__(self,in_channels) -> None:
        # CONV: conv+bn+relu
        conv1=nn.Conv2d(in_channels,in_channels*2,kernel_size=3,padding=3//2)
        bn1=nn.BatchNorm2d(in_channels*2)
        # ln1=nn.LayerNorm([in_channels*2,224,224])
        relu1= nn.ReLU(inplace=True)
        conv2=nn.Conv2d(in_channels*2,in_channels,kernel_size=3,padding=3//2)
        bn2=nn.BatchNorm2d(in_channels)
        # ln2=nn.LayerNorm([in_channels,224,224])
        relu2=nn.ReLU(inplace=True)
        conv3=nn.Conv2d(in_channels,in_channels//2,kernel_size=3,padding=3//2)
        bn3=nn.BatchNorm2d(in_channels//2)
        # ln3=nn.LayerNorm([in_channels//2,224,224])
        relu3=nn.ReLU(inplace=True)
        super().__init__(conv1,bn1,relu1,conv2,bn2,relu2,conv3,bn3,relu3)

class Conv_dice(nn.Sequential):
    def __init__(self,in_channels) -> None:
        conv1=nn.Conv2d(in_channels,in_channels*2,kernel_size=3,padding=3//2)
        bn1=nn.BatchNorm2d(in_channels*2)
        # ln1=nn.LayerNorm([in_channels*2,224,224])
        relu1= nn.ReLU(inplace=True)
        conv2=nn.Conv2d(in_channels*2,in_channels,kernel_size=3,padding=3//2)
        bn2=nn.BatchNorm2d(in_channels)
        # ln2=nn.LayerNorm([in_channels,224,224])
        relu2=nn.ReLU(inplace=True)
        conv3=nn.Conv2d(in_channels,in_channels//2,kernel_size=3,padding=3//2)
        bn3=nn.BatchNorm2d(in_channels//2)
        # ln3=nn.LayerNorm([in_channels//2,224,224])
        relu3=nn.ReLU(inplace=True)
        super().__init__(conv1,bn1,relu1,conv2,bn2,relu2,conv3,bn3,relu3)

class Conv_merge(nn.Sequential):
    def __init__(self,in_channels) -> None:
        conv1=nn.Conv2d(in_channels,in_channels*2,kernel_size=3,padding=3//2)
        bn1=nn.BatchNorm2d(in_channels*2)
        # ln1=nn.LayerNorm([in_channels*2,224,224])
        relu1= nn.ReLU(inplace=True)
        conv2=nn.Conv2d(in_channels*2,in_channels,kernel_size=3,padding=3//2)
        bn2=nn.BatchNorm2d(in_channels)
        # ln2=nn.LayerNorm([in_channels,224,224])
        relu2=nn.ReLU(inplace=True)
        conv3=nn.Conv2d(in_channels,in_channels//2,kernel_size=3,padding=3//2)
        bn3=nn.BatchNorm2d(in_channels//2)
        # ln3=nn.LayerNorm([in_channels//2,224,224])
        relu3=nn.ReLU(inplace=True)
        super().__init__(conv1,bn1,relu1,conv2,bn2,relu2,conv3,bn3,relu3)
        

class Main_Model(nn.Module):
    def __init__(self,image_size=224,num_models=2,attn_features=32,encoder_name_ls=None,with_edge=True,with_attn=True) -> None:
        super().__init__()
        self.with_edge=with_edge
        self.with_attn=with_attn
        if encoder_name_ls is None:
            encoder_name_ls=['densenet121']*num_models
        self.numbers=num_models
        self.seg_block=nn.ModuleList([Segmentation_Block(encoder_name=encoder_name_ls[i],attn_features=attn_features).cuda() for i in range(num_models)])
        if self.with_attn:
            self.attn_map=CrossAttention_Map(image_size,attn_features,num_models)
        self.conv_without_attn=nn.Conv2d(16*num_models,attn_features*2,kernel_size=3,padding=3//2)
        self.direct_segmentation=nn.Sequential(nn.Conv2d(16,4,kernel_size=3,padding=3//2))
        if self.with_edge:
            self.conv_edge=Conv_edge(16*2)
            self.seg_edge=nn.Conv2d(16,8,kernel_size=3,padding=3//2)
        self.conv_dice=Conv_dice(16*2)
        self.seg_dice=nn.Conv2d(16,4,kernel_size=3,padding=3//2)
        self.attn_ed=CrossAttention(image_size//4*image_size//4,attn_features).cuda()
        self.avgpool=nn.AvgPool2d(4)
        self.upsample=nn.UpsamplingBilinear2d(scale_factor=4)
        

        self.conv_merge=Conv_merge(16*2)
# class CrossAttention_Map(nn.Module):
#     def __init__(self,image_size,attn_features,numbers) -> None:
#         super().__init__()
#         self.numbers=numbers

#         self.attn_map=nn.ModuleDict({f"{i}_{j}":CrossAttention(image_size//4*image_size//4,attn_features).cuda() for i in range(numbers) for j in range(numbers)})
#         self.conv_re_ls=nn.ModuleList([nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=4),nn.Conv2d(attn_features*numbers,16,kernel_size=3,padding=3//2)).cuda() for _ in range(numbers)])
#         self.conv_cat=nn.Conv2d(16*numbers,16*2,kernel_size=3,padding=3//2)

    
#     def forward(self,x_ls,f_x_ls):
#         attn_x=torch.cat([self.conv_re_ls[i](torch.cat([(self.attn_map[f"{i}_{j}"](f_x_ls[i],f_x_ls[j])) for j in range(self.numbers)],dim=1))+x_ls[i] for i in range(self.numbers)],dim=1)
#         attn_x=self.conv_cat(attn_x)

#         return attn_x  #channels=32
    
    def forward(self,x):
        x_ls=[]
        pred_x_ls=[]
        f_x_ls=[]
        for i in range(self.numbers):
            x_,pred_x_,f_x_=self.seg_block[i](x)
            x_ls.append(x_)
            pred_x_ls.append(pred_x_)
            f_x_ls.append(f_x_)

        if self.numbers==1:
            pred_dice=self.direct_segmentation(torch.cat(x_ls,dim=1))
            return pred_dice

        if self.with_attn:
            attn_x=self.attn_map(x_ls,f_x_ls)  # B*16*2*H*W
        else:
            attn_x=self.conv_without_attn(torch.cat(x_ls,dim=1))

        f_dice=self.conv_dice(attn_x)
        if self.with_edge:
            f_edge=self.conv_edge(attn_x)
            # calculate attn
            pred_edge=self.seg_edge(f_edge)
            # attn_ed=self.upsample(self.attn_ed(self.avgpool(f_edge),self.avgpool(f_dice)))
            pred_dice=self.seg_dice(self.conv_merge(torch.cat([f_dice,f_edge],dim=1))+f_dice)
            # pred_dice=self.seg_dice(attn_ed+f_dice) # 残差连接
            
            return pred_x_ls,pred_edge,pred_dice
        else:
            pred_dice=self.seg_dice(f_dice)
            return pred_x_ls,pred_dice

        
class FCWCRFLoss(nn.Module):
    def __init__(self,classes=4,theta=0.5,resize=None) -> None:
        super().__init__()
        self.classes=classes
        self.theta=theta
        self.resize=resize #None or tuple or list

    
    def distance_2features(self,f1,f2):
        # f1:B*c*h(1)*w(1) f2:B*c*h*w
        # return B*h*w
        # 计算Gauss KL loss和Cos similarity
        mu1=f1.mean(dim=1)
        mu2=f2.mean(dim=1)
        std1=f1.std(dim=1)
        std2=f2.std(dim=1)
        # return std1-std2
        KL_gauss=1/2*(torch.log(std2/std1)+(std1**2+(mu1-mu2)**2)/(2*std2**2)+torch.log(std1/std2)+(std2**2+(mu1-mu2)**2)/(2*std1**2)-1) #B*1*h*w
        cos_simi=(f1*f2).sum(dim=1)/torch.sqrt((f1**2).sum(dim=1)*(f2**2).sum(dim=1)) # 


        return (KL_gauss+2*(1-cos_simi))/3

    def forward(self,f,p):
        # f:features(B*F_C*H*W) p:pred_dice(B*C*H*W)
        k_p=self.classes/torch.log(torch.tensor(self.classes)).item()
        p=F.softmax(p,dim=1)[:,1:,...]
        if self.resize is not None:
            f=F.interpolate(f,size=self.resize)
            p=F.interpolate(f,size=self.resize)
            h,w=self.resize
        else:
            h,w=p.shape[-2:]
            f=f.reshape(f.shape)
        f=f.reshape(*(f.shape[:2]),-1)
        p=p.reshape(*(p.shape[:2]),-1)
        theta=self.theta
        score=sum([(torch.exp(-self.distance_2features(f[...,ix].unsqueeze(-1),f)**2/theta**2).unsqueeze(1)*(2*p*p[...,ix].unsqueeze(-1)).abs()/(p**2+p[...,ix].unsqueeze(-1)**2)*((1+k_p*p*torch.log(p))*(1+k_p*p[...,ix]*torch.log(p[...,ix])).unsqueeze(-1))) for ix in range(h*w)]).mean()
        return 1-score/h/w


class GradientLoss(nn.Module):
    def __init__(self, operator="Sobel", channel_mean=True,loss='dice',use_opencv=False,sigma=5.0):
        r"""
       :param operator: in ['Sobel', 'Prewitt','Roberts','Scharr']
       :param channel_mean: 是否在通道维度上计算均值
       """
        super(GradientLoss, self).__init__()
        assert operator in ['Sobel', 'Prewitt', 'Roberts', 'Scharr'], "Unsupported operator"
        self.channel_mean = channel_mean
        self.operators = {
            "Sobel": {
                'x': torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float).cuda(),
                'y': torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float).cuda()
            },
            "Prewitt": {
                'x': torch.tensor([[[[1, 0, -1], [1, 0, -1], [1, 0, -1]]]], dtype=torch.float).cuda(),
                'y': torch.tensor([[[[-1, -1, -1], [0, 0, 0], [1, 1, 1]]]], dtype=torch.float).cuda()
            },
            "Roberts": {
                'x': torch.tensor([[[[1, 0], [0, -1]]]], dtype=torch.float).cuda(),
                'y': torch.tensor([[[[0, -1], [1, 0]]]], dtype=torch.float).cuda()
            },
            "Scharr": {
                'x': torch.tensor([[[[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]]], dtype=torch.float).cuda(),
                'y': torch.tensor([[[[-3, 10, -3], [0, 0, 0], [3, 10, 3]]]], dtype=torch.float).cuda()
            },
        }
        self.op_x = self.operators[operator]['x']
        self.op_y = self.operators[operator]['y']
        self.diceloss=smp.losses.DiceLoss(mode='multiclass',classes=[1,2,3])
        self.ssim_loss=SSIM(di=True)
        self.loss=loss
        self.use_opencv=False
        self.sigma=sigma
 
    def gradients(self, img_tensor):
        """
        img_tensor:n*4*h*w
        """
        grad_img=torch.zeros_like(img_tensor).detach().cpu().numpy()
        if self.use_opencv:
            img_tensor=img_tensor.detach().cpu().numpy().astype('uint8')*255
            for i in img_tensor.shape[0]:
                for k in range(4):
                    grad_img[i][k]=cv2.Canny(img_tensor[i][k],100,100)
            grad_img[grad_img>0]=1
            grad_img=torch.tensor(grad_img).cuda().float()
            return grad_img

        else:
            op_x, op_y = self.op_x, self.op_y
            if self.channel_mean:
                img_tensor = img_tensor.mean(dim=1, keepdim=True)
                groups = 1
            else:
                groups = img_tensor.shape[1]
                op_x = op_x.repeat(groups, 1, 1, 1)
                op_y = op_y.repeat(groups, 1, 1, 1)
            grad_x = F.conv2d(img_tensor, op_x, groups=groups)
            grad_y = F.conv2d(img_tensor, op_y, groups=groups)
            grad_img=grad_x.abs()+grad_y.abs()
            grad_img[grad_img>0]=1.
            return grad_img

    def boundary_loss(self,grad_gt,edge):
        # the boundary loss of class K
        # grad_gt:Tensor B*H*W ,edge:Tensor B*H*W  in Class K return the class K's loss
        B=grad_gt.shape[0]
        loss_gradK=0
        for i in range(B):
            dist_gt=torch.tensor(np.exp(-distance_transform_edt(~grad_gt[i].detach().cpu().numpy().astype('bool'))**2/(self.sigma**2))).cuda()
            loss_gradK+=1-(edge[i]*dist_gt).sum()/torch.max(edge[i].sum(),grad_gt[i].sum())
        return loss_gradK/B





 
    def forward(self, edge, img_gt): # edge:B*n*h*w  img:B*(n+1)*h*w
        img=F.one_hot(img_gt.long(),num_classes=4).permute(0,3,1,2).float()
        grad_img=self.gradients(img)
        grad_img=F.interpolate(grad_img,img.shape[-1],mode='nearest')
        # print(edge.shape,grad_img.shape)
        score=0
        loss_sum=0
        for i in range(4):
            if self.loss=='boundary':
                edge_i=F.softmax(edge[:,2*i:2*(i+1),...],dim=1)[:,1,...]
                loss_sum+=self.boundary_loss(grad_img[:,i],edge_i)
            elif self.loss=='dice':
                edge_i=torch.round(F.softmax(edge[:,2*i:2*(i+1),...],dim=1)[:,1,...].unsqueeze(1))

                # loss_sum+=(1-self.ssim_loss(edge_i,grad_img[:,i,...].unsqueeze(1)))
                loss_sum+=(1-(2*edge_i*grad_img[:,i,...]).sum()/((edge_i+grad_img[:,i,...]).sum()+1e-10))
            

        return loss_sum/4

