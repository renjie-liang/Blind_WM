import torch
import torch.nn as nn
from lib.config import cfg
from model.encoder.resnet18_dilation import resnet18_dilation


class LJJ_Decoder(nn.Module):
    """
    BASE_Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """
    def __init__(self):

        super(LJJ_Decoder, self).__init__()



        self.H_img = cfg.DATA_SET.H_IMG 
        self.W_img = cfg.DATA_SET.W_IMG
        self.H_wm = cfg.DATA_SET.H_WM
        self.W_wm = cfg.DATA_SET.W_WM
        self.Num_wm = self.H_wm * self.W_wm 
        self.batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU
        

        self.img_resnet = resnet18_dilation()

        self.wm_expend = nn.Sequential(
            nn.Conv2d(512, self.Num_wm, 1, padding=0),
            nn.BatchNorm2d(self.Num_wm),
            nn.ReLU(inplace = True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
            )

        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(self.Num_wm, self.Num_wm, bias = True)


    def forward(self, img_wm):
        wm = self.img_resnet(img_wm)
        wm = self.wm_expend(wm) # -1 * Num_wm * 1 * 1

        wm.squeeze_().squeeze_()
        wm = self.fc(wm)
        wm = wm.reshape(wm.size(0), 1, self.H_wm, self.W_wm)
        return wm
