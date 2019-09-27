import torch
import torch.nn as nn
from lib.config import cfg


class U_NET2(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self):
        super(U_NET2, self).__init__()
        self.H_img = cfg.DATA_SET.H_IMG 
        self.W_img = cfg.DATA_SET.W_IMG
        self.H_wm = cfg.DATA_SET.H_WM
        self.W_wm = cfg.DATA_SET.W_WM
        self.Num_wm = self.H_wm * self.W_wm 
        
        self.batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU

        self.wm_expend = nn.Sequential(
            nn.Conv2d(self.Num_wm, 1024, 1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace = True),
            )


        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=3,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,3,kernel_size=1,stride=1,padding=0)
        self.merge = nn.Conv2d(6,3,1,padding = 0)

        self.fc = nn.Linear(self.Num_wm, self.Num_wm, bias = True)

    def forward(self,img, wm):
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


        # decoding + concat path

        # wm = self.fc(wm)
        wm = wm.reshape(wm.size(0),-1)
        wm =wm.unsqueeze(-1).unsqueeze(-1)
        wm = wm.expand(-1,-1, 8, 8) # -1 * Num_wm * 8 * 8
        wm = self.wm_expend(wm) # -1 * 1024 * 8 * 8
        x5 =  wm
        # x5 = torch.cat((x5,wm), dim=1) # -1 * 1024+1 * 8 * 8

        d5 = self.Up5(x5)               # -1 * 512 * 16 * 16

        d5 = torch.cat((x4,d5),dim=1)

        
        d5 = self.Up_conv5(d5)
        
        
        d4 = self.Up4(d5)
        # d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        # d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        # d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        img_wm = torch.cat([d1, img],dim = 1)
        img_wm = self.merge(img_wm)


        return img_wm


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