import torch
import torch.nn as nn
from lib.config import cfg


class DeConv_Decoder_WM(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self):
        super(DeConv_Decoder_WM, self).__init__()
        self.H_img = cfg.DATA_SET.H_IMG 
        self.W_img = cfg.DATA_SET.W_IMG
        self.H_wm = cfg.DATA_SET.H_WM
        self.W_wm = cfg.DATA_SET.W_WM
        self.Num_wm = self.H_wm * self.W_wm 
        
        self.batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU
        
        img_conv_channels = cfg.HiddenNet.IMG_ENCODER_CHANNELS
        img_conv_blocks = cfg.HiddenNet.IMG_ENCODER_BLOCKS
        
        layers = []
        for _ in range(cfg.DeConv.WMConv_DECODER.WM_LAYERS_NUM):
            layers.append(nn.Conv2d(64,64,1,padding =0))
            layers.append(nn.ReLU(inplace=True))

        self.wmconv = nn.Sequential(*layers)


        self.conv1 = torch.nn.Conv2d(in_channels = 3,
                                    out_channels = 64,
                                    kernel_size = 3,
                                    stride = 1,
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
        self.down_conv4 = torch.nn.Conv2d(in_channels = 64,
                                    out_channels = 64,
                                    kernel_size = 3,
                                    stride = 2,
                                    padding =1)
        self.conv2 = torch.nn.Conv2d(in_channels = 64,
                                    out_channels = 1,
                                    kernel_size = 3,
                                    stride = 1,
                                    padding =1)

        self.relu = torch.nn.ReLU(inplace=True)
        self.sigm = torch.nn.Sigmoid()
        self.wmdeconv = torch.nn.Conv2d(64,64,3,padding =1)
        self.imgconv = torch.nn.Conv2d(64,64,3,padding =1)

    def forward(self, img_wm):
        # img_wm -1 * 3 * 128 * 128




        wm = self.conv1(img_wm) # -1 * 64 * 128 * 128
        wm = self.relu(wm) ## -1 * 64 * 128 * 128

        wm = self.imgconv(wm)
        wm = self.relu(wm) ## -1 * 64 * 128 * 128

        wm = self.wmdeconv(wm) ## -1 * 64 * 32 * 32
        wm = self.relu(wm) ## -1 * 64 * 32 * 32


        wm = self.down_conv2(wm)     # -1 * 64 * 64 * 64
        wm = self.relu(wm)  # -1 * 64 * 64 * 64
        
        wm = self.wmdeconv(wm) ## -1 * 64 * 32 * 32
        wm = self.relu(wm) ## -1 * 64 * 32 * 32

        wm = self.down_conv3(wm)     # -1 * 64 * 32 * 32
        wm = self.relu(wm) ## -1 * 64 * 32 * 32
        
        wm = self.wmdeconv(wm) ## -1 * 64 * 32 * 32
        wm = self.relu(wm) ## -1 * 64 * 32 * 32


        wm = self.down_conv4(wm)     # -1 * 64 * 16 * 16
        wm = self.relu(wm) ## -1 * 64 * 16 * 16


        wm = self.wmconv(wm) ## -1 * 64 * 16 * 16   
 
        wm = self.conv2(wm)   # -1 * 1 * 16 * 16

        # wm = self.sigm(wm)
        return wm
