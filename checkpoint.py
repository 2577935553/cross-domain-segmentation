import torch
import os
import numpy as np
import tifffile


def intermediate_output(deblur_net,device,path,min_v,max_v,epoch,iters):
    deblur_net.eval()
    device_cpu = torch.device('cpu')

    output_path = path+'checkpoint/'+'intermediate_results/'
    input_img_path=path+'raw_data/'+'xz/'

    index=1 # users define

    input_img = tifffile.imread(input_img_path+str(index)+'.tif')
    input_img = (input_img.astype(np.float32) - min_v) / (max_v - min_v)
    input_img[input_img > 1] = 1
    input_img[input_img < 0] = 0
    input_img = np.expand_dims(input_img, axis=0)
    input_img = np.expand_dims(input_img, axis=0)
    input_tensor = torch.from_numpy(input_img).to(device)
    with torch.no_grad():
        net_output = deblur_net(input_tensor)

    net_output = net_output.squeeze_(0).squeeze_(0).to(device_cpu).numpy()
    net_output = net_output * (max_v-min_v)+min_v
    net_output = np.clip(net_output, 0, max_v)
    net_output = np.array(net_output, dtype=np.uint16)

    tifffile.imwrite(output_path + 'Self_net_epoch_{}'.format(epoch + 1) + '_iters_{}_'.format(iters) + str(index)+'.tif',
                     net_output)

    deblur_net.train()




