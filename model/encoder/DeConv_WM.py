import torch
import torch.nn as nn
from lib.config import cfg


class DeConv_WM(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self):
        super(DeConv_WM, self).__init__()
        self.H_img = cfg.DATA_SET.H_IMG 
        self.W_img = cfg.DATA_SET.W_IMG
        self.H_wm = cfg.DATA_SET.H_WM
        self.W_wm = cfg.DATA_SET.W_WM
        self.Num_wm = self.H_wm * self.W_wm 
        
        self.batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU
        
        img_conv_channels = cfg.HiddenNet.IMG_ENCODER_CHANNELS
        img_conv_blocks = cfg.HiddenNet.IMG_ENCODER_BLOCKS
        

        self.wm_conv = nn.Conv2d(self.H_img//8 * self.W_img//8, 256, 1)

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
        self.conv1 = torch.nn.Conv2d(1,64,3,padding =1)
        self.conv2 = torch.nn.Conv2d(64,3,3,padding =1)
    def forward(self, img, wm):
        # wm -1 * 1 * 16 * 16
        wm = self.conv1(wm) ## -1 * 64 * 16 * 16
        wm = self.deconv1(wm) # -1 * 64 * 32 * 32
        wm = self.deconv2(wm) # -1 * 64 * 64 * 64
        wm = self.deconv3(wm) # -1 * 64 * 128 * 128
        wm = self.conv2(wm)  # -1 * 3 * 128 * 128
        return wm
