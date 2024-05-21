import common

import torch.nn as nn
import torch
import sepconv
import math

# this code is from https://github.com/yulunzhang/RCAN
# A part of the code is changed.


class Channel_Spatial_Attention_Module(nn.Module):
    def __init__(self):
        super(Channel_Spatial_Attention_Module, self).__init__()
        self.conv_3d = nn.Conv3d(1, 1, kernel_size=3, padding=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        n, c, h, w = x.size()
        x_reshape = x.reshape(n, 1, c, h, w)
        x_3d = self.sigmoid(self.conv_3d(x_reshape))
        x_squzzed = x_3d.reshape(n, c, h, w)
        return (self.scale * x_squzzed) * x + x


class Layer_Attention_Module(nn.Module):
    def __init__(self, config):
        super(Layer_Attention_Module, self).__init__()
        self.softmax = nn.Softmax(dim=2)
        self.scale = nn.Parameter(torch.zeros(1))
        self.n = config['MODEL_CONFIG']['N_RESGROUPS']
        self.c = config['MODEL_CONFIG']['N_FEATURES']
        self.conv = nn.Conv2d(self.n * self.c, self.c, kernel_size=3, padding=1)

    def forward(self, feature_group):
        b,n,c,h,w = feature_group.size()
        feature_group_reshape = feature_group.view(b, n, c * h * w)

        attention_map = torch.bmm(feature_group_reshape, feature_group_reshape.view(b, c * h * w, n))
        attention_map = self.softmax(attention_map) # N * N

        attention_feature = torch.bmm(attention_map, feature_group_reshape) # N * CHW
        b, n, chw = attention_feature.size()
        attention_feature = attention_feature.view(b,n,c,h,w)

        attention_feature = self.scale * attention_feature + feature_group
        b, n, c, h, w = attention_feature.size()
        return self.conv(attention_feature.view(b, n * c, h, w))



## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = list()
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        return res + x


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, config, conv=common.default_conv):
        super(RCAN, self).__init__()

        n_resgroups = config['MODEL_CONFIG']['N_RESGROUPS']
        n_resblocks = config['MODEL_CONFIG']['N_RESBLOCKS']
        n_feats = config['MODEL_CONFIG']['N_FEATURES']
        kernel_size = 3
        reduction = config['MODEL_CONFIG']['REDUCTION']
        scale = config['MODEL_CONFIG']['UP_SCALE']
        act = nn.ReLU(True)

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        #self.sub_mean = common.MeanShift(config['MODEL_CONFIG']['RGB_RANGE'], rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(config['MODEL_CONFIG']['N_COLORS'], n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=config['MODEL_CONFIG']['RES_SCALE'], n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(2*n_feats, config['MODEL_CONFIG']['N_COLORS'], kernel_size)]

        #self.add_mean = common.MeanShift(config['MODEL_CONFIG']['RGB_RANGE'], rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.ModuleList(modules_body)
        self.tail = nn.Sequential(*modules_tail)
        self.CSA = Channel_Spatial_Attention_Module()
        self.LA = Layer_Attention_Module(config)

    def forward(self, x):
        #x = self.sub_mean(x)
        x = self.head(x)

        body_results = list()
        body_results.append(x)
        for RG in self.body:
            x = RG(x)
            body_results.append(x)

        feature_LA = self.LA(torch.stack(body_results[1:-1], dim=1)) # b, n * c, h, w
        feature_CSA = self.CSA(body_results[-1]) # # b, c, h, w

        x = self.tail(body_results[0] + feature_CSA + feature_LA)
        #x = self.add_mean(x)

        del body_results

        return x
    
class SSMM(torch.nn.Module): # sepconv slomo master model
    def __init__(self,color_channels=1):
        super().__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )
        # end

        def Upsample(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )
        # end

        def Subnet():
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=51, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=51, out_channels=51, kernel_size=3, stride=1, padding=1)
            )
        # end

        self.netConv1 = Basic(2*color_channels, 32)
        self.netConv2 = Basic(32, 64)
        self.netConv3 = Basic(64, 128)
        self.netConv4 = Basic(128, 256)
        self.netConv5 = Basic(256, 512)

        self.netDeconv5 = Basic(512, 512)
        self.netDeconv4 = Basic(512, 256)
        self.netDeconv3 = Basic(256, 128)
        self.netDeconv2 = Basic(128, 64)

        self.netUpsample5 = Upsample(512, 512)
        self.netUpsample4 = Upsample(256, 256)
        self.netUpsample3 = Upsample(128, 128)
        self.netUpsample2 = Upsample(64, 64)

        self.netVertical1 = Subnet()
        self.netVertical2 = Subnet()
        self.netHorizontal1 = Subnet()
        self.netHorizontal2 = Subnet()
# 
        # self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/sepconv/network-' + args_strModel + '.pytorch', file_name='sepconv-' + args_strModel).items() })
    # end

    def forward(self, tenOne, tenTwo):
        tenConv1 = self.netConv1(torch.cat([tenOne, tenTwo], 1))
        tenConv2 = self.netConv2(torch.nn.functional.avg_pool2d(input=tenConv1, kernel_size=2, stride=2, count_include_pad=False))
        tenConv3 = self.netConv3(torch.nn.functional.avg_pool2d(input=tenConv2, kernel_size=2, stride=2, count_include_pad=False))
        tenConv4 = self.netConv4(torch.nn.functional.avg_pool2d(input=tenConv3, kernel_size=2, stride=2, count_include_pad=False))
        tenConv5 = self.netConv5(torch.nn.functional.avg_pool2d(input=tenConv4, kernel_size=2, stride=2, count_include_pad=False))

        tenDeconv5 = self.netUpsample5(self.netDeconv5(torch.nn.functional.avg_pool2d(input=tenConv5, kernel_size=2, stride=2, count_include_pad=False)))
        tenDeconv4 = self.netUpsample4(self.netDeconv4(tenDeconv5 + tenConv5))
        tenDeconv3 = self.netUpsample3(self.netDeconv3(tenDeconv4 + tenConv4))
        tenDeconv2 = self.netUpsample2(self.netDeconv2(tenDeconv3 + tenConv3))

        tenCombine = tenDeconv2 + tenConv2

        tenVerone = self.netVertical1(tenCombine)
        tenVertwo = self.netVertical2(tenCombine)
        tenHorone = self.netHorizontal1(tenCombine)
        tenHortwo = self.netHorizontal2(tenCombine)

        return sum([
            sepconv.sepconv_func.apply(torch.nn.functional.pad(input=tenOne, pad=[int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0))], mode='replicate'), tenVerone, tenHorone),
            sepconv.sepconv_func.apply(torch.nn.functional.pad(input=tenTwo, pad=[int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0))], mode='replicate'), tenVertwo, tenHortwo)
        ])
    # end
# end

netNetwork = None