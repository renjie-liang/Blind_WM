import torch
import torch.nn as nn
from lib.config import cfg


class DeConv_WMFlatten_Decoder(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self):
        super(DeConv_WMFlatten_Decoder, self).__init__()
        self.H_img = cfg.DATA_SET.H_IMG 
        self.W_img = cfg.DATA_SET.W_IMG
        self.H_wm = cfg.DATA_SET.H_WM
        self.W_wm = cfg.DATA_SET.W_WM
        self.Num_wm = self.H_wm * self.W_wm 
        
        self.batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU
        
        img_conv_channels = cfg.HiddenNet.IMG_ENCODER_CHANNELS
        img_conv_blocks = cfg.HiddenNet.IMG_ENCODER_BLOCKS
        
        self.imgconv = torch.nn.Conv2d(3,64,3,padding =1)
        self.extr = nn.Conv2d(64,64,1,padding =0)



        self.down_conv1 = torch.nn.Conv2d(in_channels = 64,
                                    out_channels = 64,
                                    kernel_size = 3,
                                    stride = 2,
                                    padding =1)
        self.down_conv2 = torch.nn.Conv2d(in_channels = 64,
                                    out_channels = 64,
                                    kernel_size = 3,
                                    stride = 2,
                                    padding =1)
        self.down_conv3 = torch.nn.Conv2d(in_channels = 64,
                                    out_channels = 64,
                                    kernel_size = 3,
                                    stride = 2,
                                    padding =1)

        self.flatten_conv = torch.nn.Conv2d(64,self.Num_wm , 1, padding =0)
        self.adpavgpool = torch.nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.relu = torch.nn.ReLU(inplace=True)
        self.wmdeconv = torch.nn.Conv2d(64,64,3,padding =1)
        self.imgconv2 = torch.nn.Conv2d(64,64,3,padding =1)

        self.bn64 = nn.BatchNorm2d(64)

    def forward(self, img_wm):
        # img_wm -1 * 3 * 128 * 128

        img_r = self.imgconv(img_wm) # -1 * 64 * 128 * 128
        img_r = self.bn64(img_r)

        img_r = self.relu(img_r) ## -1 * 64 * 128 * 128

        ### di ###
        img_r = self.imgconv2(img_r) # -1 * 64 * 128 * 128
        img_r = self.bn64(img_r)
        img_r = self.relu(img_r) ## -1 * 64 * 128 * 128



        ### di ###

        wm = self.extr(img_r)
        wm = self.bn64(wm)


        wm = self.down_conv1(wm) # -1 * 64 * 64 * 64
        wm = self.bn64(wm)
        wm = self.relu(wm) 
        wm = self.wmdeconv(wm)
        wm = self.bn64(wm)
        wm = self.relu(wm) 

        wm = self.down_conv2(wm)  ## -1 * 64 * 32 * 32
        wm = self.bn64(wm)
        wm = self.relu(wm) 
        wm = self.wmdeconv(wm) 
        wm = self.bn64(wm)
        wm = self.relu(wm) 

        wm = self.down_conv3(wm) # -1 * 64 * 16 * 16
        wm = self.bn64(wm)
        wm = self.relu(wm)

        ### dw ###
        wm = self.wmdeconv(wm) 
        wm = self.bn64(wm)
        wm = self.relu(wm) 



        ### dw ###

        wm = self.flatten_conv(wm) # -1 * Num_wm  * 16 * 16
        # wm = self.relu(wm) 

        wm = self.adpavgpool(wm) # -1 * Num_wm   * 1 * 1
        wm = wm.reshape(self.batch_size, 1, self.H_wm, self.W_wm)

        return wm
