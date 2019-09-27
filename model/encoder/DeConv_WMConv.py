import torch
import torch.nn as nn
from lib.config import cfg


class DeConv_WMConv(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self):
        super(DeConv_WMConv, self).__init__()
        self.H_img = cfg.DATA_SET.H_IMG 
        self.W_img = cfg.DATA_SET.W_IMG
        self.H_wm = cfg.DATA_SET.H_WM
        self.W_wm = cfg.DATA_SET.W_WM
        self.Num_wm = self.H_wm * self.W_wm 
        
        self.batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU
        
        img_conv_channels = cfg.HiddenNet.IMG_ENCODER_CHANNELS
        img_conv_blocks = cfg.HiddenNet.IMG_ENCODER_BLOCKS
        


        self.conv1 = torch.nn.Conv2d(1,64,3,padding =1)
        layers = []
        for _ in range(cfg.DeConv.WMConv_ENCODER.WM_LAYERS_NUM):
            layers.append(nn.Conv2d(64,64,3,padding =1))
            layers.append(nn.ReLU(inplace=True))

        self.wmconv = nn.Sequential(*layers)

        self.relu = torch.nn.ReLU(inplace=True)

        
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
        self.merge = torch.nn.Conv2d(64+3,3,3,padding =1)
        self.wmdeconv = torch.nn.Conv2d(64,64,3,padding =1)
      

    def forward(self, img, wm):
        # wm -1 * 1 * 16 * 16
        # img = conv
        wm = self.conv1(wm) ## -1 * 64 * 16 * 16
        wm = self.relu(wm) ## -1 * 64 * 16 * 16

        wm = self.wmconv(wm) ## -1 * 64 * 16 * 16   
        
        wm = self.deconv1(wm) # -1 * 64 * 32 * 32
        wm = self.relu(wm) ## -1 * 64 * 32 * 32

        # wm = self.wmdeconv(wm) ## -1 * 64 * 32 * 32
        # wm = self.relu(wm) ## -1 * 64 * 32 * 32

        wm = self.deconv2(wm) # -1 * 64 * 64 * 64
        wm = self.relu(wm) ## -1 * 64 * 64 * 64

        # wm = self.wmdeconv(wm) ## -1 * 64 * 64 * 64
        # wm = self.relu(wm) ## -1 * 64 * 64 * 64


        wm = self.deconv3(wm) # -1 * 64 * 128 * 128
        wm = self.relu(wm) ## -1 * 64 * 128 * 128

        # wm = self.wmdeconv(wm) ## -1 * 64 * 128 * 128
        # wm = self.relu(wm) ## -1 * 64 * 128 * 128


        img_wm = torch.cat([wm,img],dim =1)
        img_wm = self.merge(img_wm)  # -1 * 3 * 128 * 128
        return img_wm
