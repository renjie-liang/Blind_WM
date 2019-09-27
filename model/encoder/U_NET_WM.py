import torch
import torch.nn as nn
from lib.config import cfg


class U_NET_WM(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self):
        super(U_NET_WM, self).__init__()
        self.H_img = cfg.DATA_SET.H_IMG 
        self.W_img = cfg.DATA_SET.W_IMG
        self.H_wm = cfg.DATA_SET.H_WM
        self.W_wm = cfg.DATA_SET.W_WM
        self.Num_wm = self.H_wm * self.W_wm 
        
        self.batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU
        
        img_conv_channels = cfg.HiddenNet.IMG_ENCODER_CHANNELS
        img_conv_blocks = cfg.HiddenNet.IMG_ENCODER_BLOCKS
        

        self.wm_conv = nn.Conv2d(self.H_img//8 * self.W_img//8, 256, 1)

       
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)
        self.dconv_dowm45 = double_conv(512, 1024)
        self.dconv_down5 = double_conv(self.Num_wm + 1024, 1024)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up4 = double_conv(1024 + 512 , 512)
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, 3, 1)
        
    def forward(self, img, wm):
        conv1 = self.dconv_down1(img) # -1 * 64 * 128 * 128
        x = self.maxpool(conv1)  # -1 * 64 * 64 * 64

        conv2 = self.dconv_down2(x) #-1 * 128 * 64 * 64
        x = self.maxpool(conv2) #-1 * 128 * 32 * 32
        
        conv3 = self.dconv_down3(x) # -1 * 256 * 32 * 32
        x = self.maxpool(conv3) # -1 * 256 * 16 * 16
        
        conv4 = self.dconv_down4(x) # -1 * 512 * 16 * 16
        x = self.maxpool(conv4)  # -1 * 512 * 8 * 8

        x = self.dconv_dowm45(x) # -1 * 1024 * 8 * 8
        
        wm = wm.reshape(self.batch_size,-1)
        wm =wm.unsqueeze(-1).unsqueeze(-1)
        wm = wm.expand(-1,-1, 8, 8) # -1 * Num_wm * 8 * 8
        
        img_wm = torch.cat([x, wm], dim=1)  # -1 * Num_wm + 1024 * 8 * 8
        img_wm = self.dconv_down5(img_wm)  # -1 * 1024 * 8 * 8

        x = self.upsample(x)  # -1 * 1024 * 16 * 16  
        x = torch.cat([x, conv4], dim=1) # -1 * 1024+512 * 16 * 16 
        x = self.dconv_up4(x) # -1 * 512 * 16 * 16 

        x = self.upsample(x)  # -1 * 512 * 32 * 32
        x = torch.cat([x, conv3], dim=1) # -1 * 512+256 * 32 * 32
        x = self.dconv_up3(x) # -1 * 256 * 32 * 32

        x = self.upsample(x) # -1 * 256 * 64 * 64
        x = torch.cat([x, conv2], dim=1) # -1 * 256+128* 64 * 64
        x = self.dconv_up2(x) # -1 * 128* 64 * 64

        x = self.upsample(x)  # -1 * 128 * 128 * 128
        x = torch.cat([x, conv1], dim=1)  # -1 * 128+64 * 128 * 128
        x = self.dconv_up1(x) # -1 * 64 * 128 * 128
        
        out = self.conv_last(x) # -1 * 3 * 128 * 128
        return out



def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   
