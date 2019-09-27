import torch
import torch.nn as nn
from lib.config import cfg


class BASE_Decoder(nn.Module):
    """
    BASE_Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """
    def __init__(self):

        super(BASE_Decoder, self).__init__()
        decoder_channels = cfg.BASE_Decoder.DECODER_CHANNELS
        decoder_blocks = cfg.BASE_Decoder.DECODER_BLOCKS=7
        self.batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU

        self.H_wm = cfg.DATA_SET.H_WM
        self.W_wm = cfg.DATA_SET.W_WM
        self.Num_wm = self.H_wm * self.W_wm 

        self.H_img = cfg.DATA_SET.H_IMG 
        self.W_img = cfg.DATA_SET.W_IMG
        self.Num_img = self.H_img * self.W_img 

        layers = [ConvBNRelu(3, decoder_channels)]
        for _ in range(decoder_blocks - 1):
            layers.append(ConvBNRelu(decoder_channels, decoder_channels))

        layers.append(ConvBNRelu(decoder_channels,self.Num_wm ))
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        
        self.layers = nn.Sequential(*layers)
        
        # self.avg_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(in_features = self.Num_wm, 
                             out_features = self.Num_wm , 
                             bias=True)
        

    def forward(self, img_wm):
        x = self.layers(img_wm)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        #x.squeeze_(3).squeeze_(2)

        x = x.reshape(self.batch_size, -1)
        x = self.fc(x)
        x = x.reshape(self.batch_size, 1, self.H_wm, self.W_wm)

        return x


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
