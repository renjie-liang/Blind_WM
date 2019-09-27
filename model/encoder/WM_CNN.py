import torch
import torch.nn as nn
from lib.config import cfg


class WM_CNN(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self):
        super(WM_CNN, self).__init__()
        self.H_img = cfg.DATA_SET.H_IMG 
        self.W_img = cfg.DATA_SET.W_IMG
        self.H_wm = cfg.DATA_SET.H_WM
        self.W_wm = cfg.DATA_SET.W_WM
        self.Num_wm = self.H_wm * self.W_wm 
        
        self.batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU
        
        img_conv_channels = cfg.HiddenNet.IMG_ENCODER_CHANNELS
        img_conv_blocks = cfg.HiddenNet.IMG_ENCODER_BLOCKS
        

        img_encoder_layers = [ConvBNRelu(self.Num_wm, img_conv_channels)]
        # for _ in range(img_conv_blocks-1):
        #     layer = ConvBNRelu(img_conv_channels, img_conv_channels)
        #     img_encoder_layers.append(layer)

        for _ in range(2):
            layer  = nn.Sequential(
                nn.ConvTranspose2d(in_channels = img_conv_channels,
                    out_channels = img_conv_channels,
                    kernel_size = 3,
                    stride = 2,
                    padding = 1,
                    output_padding = 1),
                nn.BatchNorm2d(img_conv_channels),
                nn.ReLU(inplace=True))

            img_encoder_layers.append(layer)


        self.img_encoder_convs = nn.Sequential(*img_encoder_layers)


      
        self.after_concat_layer = ConvBNRelu(img_conv_channels + 3,
                                             img_conv_channels)
        self.final_layer = nn.Conv2d(img_conv_channels, 3, kernel_size=1,stride = 1)

    def forward(self, img, wm):
        
        wm = wm.reshape(self.batch_size,-1)
        wm =wm.unsqueeze(-1).unsqueeze(-1)
        wm = wm.expand(-1,-1, 32, 32)

        wm_conv = self.img_encoder_convs(wm)

        img_wm = torch.cat([wm_conv, img], dim=1)
        

        img_wm = self.after_concat_layer(img_wm)
        img_wm = self.final_layer(img_wm)
        

        return img_wm


class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """
    def __init__(self, channels_in, channels_out, stride=1):

        super(ConvBNRelu, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)
