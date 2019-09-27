import torch
import torch.nn as nn
from lib.config import cfg


class DeConv_WMFlatten(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self):
        super(DeConv_WMFlatten, self).__init__()
        self.H_img = cfg.DATA_SET.H_IMG 
        self.W_img = cfg.DATA_SET.W_IMG
        self.H_wm = cfg.DATA_SET.H_WM
        self.W_wm = cfg.DATA_SET.W_WM
        self.Num_wm = self.H_wm * self.W_wm 
        
        self.batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU
        
        img_conv_channels = cfg.HiddenNet.IMG_ENCODER_CHANNELS
        img_conv_blocks = cfg.HiddenNet.IMG_ENCODER_BLOCKS
        


        # self.flatten_conv = torch.nn.Conv2d(self.Num_wm,64, cfg.DeConv.WMFlatten.KERNEL_SIZE ,padding = cfg.DeConv.WMFlatten.KERNEL_SIZE // 2)
        self.flatten_conv = torch.nn.Conv2d(self.Num_wm, 64, 1 ,padding = 0)
        self.wmconv = torch.nn.Conv2d(64,64,3,padding =1)
        
        self.deconv1 = torch.nn.ConvTranspose2d(in_channels = 64, out_channels = 64,
                                kernel_size = 3, 
                                stride=2, padding=1, output_padding=1, 
                                groups=1, bias=True, dilation=1, 
                                padding_mode='zeros')
        self.deconv2 = torch.nn.ConvTranspose2d(in_channels = 64, out_channels = 64,
                                kernel_size = 3, 
                                stride=2, padding=1, output_padding=1, 
                                groups=1, bias=True, dilation=1, 
                                padding_mode='zeros')
        self.deconv3 = torch.nn.ConvTranspose2d(in_channels = 64, out_channels = 64,
                                kernel_size = 3, 
                                stride=2, padding=1, output_padding=1, 
                                groups=1, bias=True, dilation=1, 
                                padding_mode='zeros')
        self.relu = torch.nn.ReLU(inplace=True)
        self.wmdeconv = torch.nn.Conv2d(64,64,3,padding =1)

        self.imgconv = torch.nn.Conv2d(3,64,3,padding =1)
        self.imgconv2 = torch.nn.Conv2d(64,64,3,padding =1)

        self.merge_img_r_wm = torch.nn.Conv2d(128 , 3, 3,padding =1)
        self.merge = torch.nn.Conv2d(64 + 64 + 3,3,1,padding =0)
        self.bn64 = nn.BatchNorm2d(64)

    def forward(self, img, wm):
        # wm -1 * 1 * 16 * 16
        # flatten to 64 channel
        wm = wm.reshape(self.batch_size,-1) # -1 * 1 * 16 * 16
        wm =wm.unsqueeze(-1).unsqueeze(-1) # -1 * 256 * 1 * 1
        wm = wm.expand(-1,-1, 16, 16) # -1 * 256 * 16 * 16
        wm = self.flatten_conv(wm) ## -1 * 64 * 16 * 16
        wm = self.bn64(wm)


        ### ew ###
        wm = self.wmconv(wm) ## -1 * 64 * 16 * 16   
        wm = self.bn64(wm)
        wm = self.relu(wm) ## -1 * 64 * 32 * 32

        wm = self.wmconv(wm) ## -1 * 64 * 16 * 16   
        wm = self.bn64(wm)
        wm = self.relu(wm) ## -1 * 64 * 32 * 32
        wm = self.wmconv(wm) ## -1 * 64 * 16 * 16   
        wm = self.bn64(wm)
        wm = self.relu(wm) ## -1 * 64 * 32 * 32

        
        ### ew ###



        wm = self.deconv1(wm) # -1 * 64 * 32 * 32
        wm = self.bn64(wm)
        wm = self.relu(wm) ## -1 * 64 * 32 * 32

        wm = self.wmdeconv(wm) ## -1 * 64 * 32 * 32
        wm = self.bn64(wm)
        wm = self.relu(wm) ## -1 * 64 * 32 * 32

        wm = self.deconv2(wm) # -1 * 64 * 64 * 64
        wm = self.bn64(wm)
        wm = self.relu(wm) ## -1 * 64 * 64 * 64

        wm = self.wmdeconv(wm) ## -1 * 64 * 64 * 64
        wm = self.bn64(wm)
        wm = self.relu(wm) ## -1 * 64 * 64 * 64


        wm = self.deconv3(wm) # -1 * 64 * 128 * 128
        wm = self.bn64(wm)
        wm = self.relu(wm) ## -1 * 64 * 128 * 128

        wm = self.wmdeconv(wm) ## -1 * 64 * 128 * 128
        wm = self.bn64(wm)
        wm = self.relu(wm) ## -1 * 64 * 128 * 128

        img_r = self.imgconv(img) # -1 * 64 * 128 * 128
        img_r = self.bn64(img_r)
        img_r_1 = self.relu(img_r) ## -1 * 64 * 128 * 128


        ### ei ###
        img_r_2 = self.imgconv2(img_r_1) # -1 * 64 * 128 * 128
        img_r_2 = self.bn64(img_r_2)
        img_r_2 = self.relu(img_r_2) ## -1 * 64 * 128 * 128

        img_r_2 = self.imgconv2(img_r_2) # -1 * 64 * 128 * 128
        img_r_2 = self.bn64(img_r_2)
        img_r_2 = self.relu(img_r_2) ## -1 * 64 * 128 * 128

        img_r_2 = self.imgconv2(img_r_2) # -1 * 64 * 128 * 128
        img_r_2 = self.bn64(img_r_2)
        img_r_2 = self.relu(img_r_2) ## -1 * 64 * 128 * 128

        ### ei ###

        # merge_3
        img_r = img_r_1  + img_r_2
        img_wm = torch.cat([img_r, wm, img],dim =1) # -1 * 67 * 128 * 128
        img_wm = self.merge(img_wm)  # -1 * 3 * 128 * 128




        return img_wm

        ## merge_1
        # img_r = torch.cat([wm, img_r],dim =1) # -1 * 128 * 128 * 128
        # img_r = self.merge_img_r_wm(img_r) # -1 * 64 * 128 * 128
        # img_wm = torch.cat([img_r, img],dim =1) # -1 * 67 * 128 * 128
        # img_wm = self.merge(img_wm)  # -1 * 3 * 128 * 128

        ## merge_2
        # img_r = img_r * wm
        # img_r = self.merge(img_r)
        # img_wm = img + img_r

        ## merge_3
        # img_wm = torch.cat([img_r,wm, img],dim =1) # -1 * 67 * 128 * 128
        # img_wm = self.merge(img_wm)  # -1 * 3 * 128 * 128



        # merge_4 = merge_1 + residual
        # img_r_wm = torch.cat([wm, img_r],dim =1) # -1 * 128 * 128 * 128
        # img_r_wm = self.merge_img_r_wm(img_r_wm) # -1 * 64 * 128 * 128
        # img_wm = torch.cat([img_r, img, img_r_wm],dim =1) # -1 * 67 * 128 * 128
        # img_wm = self.merge(img_wm)  # -1 * 3 * 128 * 128
