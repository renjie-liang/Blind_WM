import torch
import torch.nn as nn
from lib.config import cfg


class U_NET_Half_Decoder(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self):
        super(U_NET_Half_Decoder, self).__init__()
        self.H_img = cfg.DATA_SET.H_IMG 
        self.W_img = cfg.DATA_SET.W_IMG
        self.H_wm = cfg.DATA_SET.H_WM
        self.W_wm = cfg.DATA_SET.W_WM
        self.Num_wm = self.H_wm * self.W_wm 
        
        self.batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU


        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=3,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=64)
        self.Conv3 = conv_block(ch_in=64,ch_out=64)
        self.Conv4 = conv_block(ch_in=64,ch_out=64)
        self.Conv5 = conv_block(ch_in=64,ch_out=64)



        self.wm_expend = nn.Sequential(
            nn.Conv2d(64, self.Num_wm, 1, padding=0),
            nn.BatchNorm2d(self.Num_wm),
            nn.ReLU(inplace = True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
            )
        self.fc = nn.Linear(self.Num_wm, self.Num_wm, bias = True)

    def forward(self,img):
        # encoding path
        x1 = self.Conv1(img)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)


        wm = self.wm_expend(x5) # -1 * Num_wm * 1 * 1
        # wm.squeeze_().squeeze_()
        # wm = self.fc(wm)
        wm = wm.reshape(wm.size(0), 1, self.H_wm, self.W_wm)


        return wm


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x



class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x