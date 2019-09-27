import torch
import torch.nn as nn
from lib.config import cfg


class U_Net_Half(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self):
        super(U_Net_Half, self).__init__()
        self.H_img = cfg.DATA_SET.H_IMG 
        self.W_img = cfg.DATA_SET.W_IMG
        self.H_wm = cfg.DATA_SET.H_WM
        self.W_wm = cfg.DATA_SET.W_WM
        self.Num_wm = self.H_wm * self.W_wm 
        
        self.batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU

        self.flatten_conv = torch.nn.Conv2d(self.Num_wm, 64, 1 ,padding = 0)

        self.Up4 = up_conv(ch_in=64,ch_out=64)
        self.Up_conv4 = conv_block(ch_in=64, ch_out=64)
        
        self.Up3 = up_conv(ch_in=64,ch_out=64)
        self.Up_conv3 = conv_block(ch_in=64, ch_out=64)
        
        self.Up2 = up_conv(ch_in=64,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,3,kernel_size=1,stride=1,padding=0)
        self.img_conv1 = conv_block(ch_in=3, ch_out=64)

        self.merge = nn.Conv2d(131,3,1,padding = 0)
    def forward(self, img, wm):
        # wm -1 * 1 * 16 * 16
        # flatten to 64 channel
        wm = wm.reshape(wm.size(0),-1) # -1 * 1 * 16 * 16
        wm =wm.unsqueeze(-1).unsqueeze(-1) # -1 * 256 * 1 * 1
        wm = wm.expand(-1,-1, 16, 16) # -1 * 256 * 16 * 16
        wm = self.flatten_conv(wm) ## -1 * 64 * 16 * 16
        # wm = self.bn64(wm) #???


        ### ew ###
        # wm = self.wmconv(wm) ## -1 * 64 * 16 * 16   
        # wm = self.bn64(wm)
        # wm = self.relu(wm) #

        # wm = self.wmconv(wm) ## -1 * 64 * 16 * 16   
        # wm = self.relu(wm) ## -1

        # wm = self.wmconv(wm) ## -1 * 64 * 16 * 16   
        # wm = self.relu(wm) ## -
        
        ### ew ###


        d4 = self.Up4(wm) # -1 * 64 * 32 * 32
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = self.Up_conv2(d2)

        # d1 = self.Conv_1x1(d2)



        img_r_1 = self.img_conv1(img) # -1 * 64 * 128 * 128

        ### ei ###

        # img_r = self.img_conv1(img_r_1) # -1 * 64 * 128 * 128

        ### ei ###

        # merge_3
        img_r = img_r_1 # + img_r_2
        img_wm = torch.cat([img_r, d2, img],dim =1) # -1 * 67 * 128 * 128
        img_wm = self.merge(img_wm)  # -1 * 3 * 128 * 128



        return img_wm

        ## merge_1
        # img_r = torch.cat([wm, img_r],dim =1) # -1 * 128 * 128 * 128
        # img_r = self.merge_img_r_wm(img_r) # -1 * 64 * 128 * 128
        # img_wm = torch.cat([img_r, img],dim =1) # -1 * 67 * 128 * 128
        # img_wm = self.merge(img_wm)  # -1 * 3 * 128 * 128

        ## merge_2
        # img_r = img_r * wm
        # img_r = self.merge(img_r)
        # img_wm = img + img_r

        ## merge_3
        # img_wm = torch.cat([img_r,wm, img],dim =1) # -1 * 67 * 128 * 128
        # img_wm = self.merge(img_wm)  # -1 * 3 * 128 * 128



        # merge_4 = merge_1 + residual
        # img_r_wm = torch.cat([wm, img_r],dim =1) # -1 * 128 * 128 * 128
        # img_r_wm = self.merge_img_r_wm(img_r_wm) # -1 * 64 * 128 * 128
        # img_wm = torch.cat([img_r, img, img_r_wm],dim =1) # -1 * 67 * 128 * 128
        # img_wm = self.merge(img_wm)  # -1 * 3 * 128 * 128

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