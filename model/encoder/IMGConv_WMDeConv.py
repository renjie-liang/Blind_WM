import torch
import torch.nn as nn
from lib.config import cfg


class IMGConv_WMDeConv(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self):
        super(IMGConv_WMDeConv, self).__init__()
        self.H_img = cfg.DATA_SET.H_IMG 
        self.W_img = cfg.DATA_SET.W_IMG
        self.H_wm = cfg.DATA_SET.H_WM
        self.W_wm = cfg.DATA_SET.W_WM
        self.Num_wm = self.H_wm * self.W_wm 
        
        self.batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU
        
        img_conv_channels = cfg.HiddenNet.IMG_ENCODER_CHANNELS
        img_conv_blocks = cfg.HiddenNet.IMG_ENCODER_BLOCKS
        


        layers = []
        for _ in range(cfg.DeConv.WMConv_ENCODER.WM_LAYERS_NUM):
            layers.append(nn.Conv2d(64,64,3,padding =1))
            layers.append(nn.ReLU(inplace=True))

        self.wmconv = nn.Sequential(*layers)


        self.wmconv1 = nn.Conv2d(64,64,3,padding =1)

        self.relu = torch.nn.ReLU(inplace=True)

        self.flatten_conv = torch.nn.Conv2d(self.Num_wm, 64, 1 ,padding = 0)
        
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
        self.wmdeconv = torch.nn.Conv2d(64,64,3,padding =1)
      
        self.imgconv = torch.nn.Conv2d(3,64,3,padding =1)
        self.imgconv2 = torch.nn.Conv2d(64,64,3,padding =1)

        self.merge_img_r_wm = torch.nn.Conv2d(128 , 64, 3,padding =1)
        self.merge1 = torch.nn.Conv2d(128+3, 64 ,1,padding =0)
        self.merge2 = torch.nn.Conv2d(64, 64, 3,padding =1)
        self.merge3 = torch.nn.Conv2d(64, 3, 3,padding =1)
        self.merge = torch.nn.Conv2d(64 + 3,3,1,padding =0)

        self.expand_wm = torch.nn.Conv2d(1,64,3,padding =1)

    def forward(self, img, wm):


        # conv to 64 channel
        wm = self.expand_wm(wm) ## -1 * 64 * 16 * 16
        wm = self.relu(wm) ## -1 * 64 * 16 * 16

        wm = self.wmconv1(wm) ## -1 * 64 * 16 * 16   
        wm = self.relu(wm) ## -1 * 64 * 16 * 16

        wm = self.wmconv1(wm) ## -1 * 64 * 16 * 16   
        wm = self.relu(wm) ## -1 * 64 * 16 * 16

        wm = self.wmconv1(wm) ## -1 * 64 * 16 * 16   
        wm = self.relu(wm) ## -1 * 64 * 16 * 16
        



        wm = self.deconv1(wm) # -1 * 64 * 32 * 32
        wm = self.relu(wm) ## -1 * 64 * 32 * 32

        wm = self.wmdeconv(wm) ## -1 * 64 * 32 * 32
        wm = self.relu(wm) ## -1 * 64 * 32 * 32

        wm = self.deconv2(wm) # -1 * 64 * 64 * 64
        wm = self.relu(wm) ## -1 * 64 * 64 * 64

        wm = self.wmdeconv(wm) ## -1 * 64 * 64 * 64
        wm = self.relu(wm) ## -1 * 64 * 64 * 64


        wm = self.deconv3(wm) # -1 * 64 * 128 * 128
        wm = self.relu(wm) ## -1 * 64 * 128 * 128

        wm = self.wmdeconv(wm) ## -1 * 64 * 128 * 128
        wm = self.relu(wm) ## -1 * 64 * 128 * 128

        img_r = self.imgconv(img) # -1 * 64 * 128 * 128
        img_r = self.relu(img_r) ## -1 * 64 * 128 * 128

        # img_r = self.imgconv2(img_r) # -1 * 64 * 128 * 128
        # img_r = self.relu(img_r) ## -1 * 64 * 128 * 128

        # img_r = self.imgconv2(img_r) # -1 * 64 * 128 * 128
        # img_r = self.relu(img_r) ## -1 * 64 * 128 * 128





        img_r = wm * img_r

        
        # img_r = torch.cat([wm, img_r],dim =1) # -1 * 128 * 128 * 128
        # img_r = self.merge_img_r_wm(img_r) # -1 * 64 * 128 * 128

        img_wm = torch.cat([img_r, img],dim =1) # -1 * 67 * 128 * 128
        img_wm = self.merge(img_wm)  # -1 * 3 * 128 * 128


        return img_wm






        # img_wm = torch.cat([wm, img_r, img],dim =1) # -1 * 128 + 3 * 128 * 128


        # img_wm = self.merge1(img_wm)  # -1 * 64 * 128 * 128
        # img_wm = self.relu(img_wm) ## -1 * 64 * 128 * 128
        # img_wm = self.merge2(img_wm)  # -1 * 64 * 128 * 128
        # img_wm = self.relu(img_wm) ## -1 * 64 * 128 * 128
        # img_wm = self.merge3(img_wm)  # -1 * 3 * 128 * 128





        # img_r = torch.cat([wm, img_r],dim =1) # -1 * 128 * 128 * 128
        # img_r = self.merge_img_r_wm(img_r) # -1 * 64 * 128 * 128

        # img_wm = img + img_r * 0.1
