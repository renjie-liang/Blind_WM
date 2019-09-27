import torch
import torch.nn as nn
from lib.config import cfg
from model.encoder.resnet18_dilation import resnet18_dilation

class LJJ_Net(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self):
        super(LJJ_Net, self).__init__()



        self.H_img = cfg.DATA_SET.H_IMG 
        self.W_img = cfg.DATA_SET.W_IMG
        self.H_wm = cfg.DATA_SET.H_WM
        self.W_wm = cfg.DATA_SET.W_WM
        self.Num_wm = self.H_wm * self.W_wm 
        self.batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU
        

        self.img_resnet = resnet18_dilation()

        self.img_expend = nn.Sequential(
            nn.Conv2d(512, self.Num_wm, 1, padding=0),
            nn.BatchNorm2d(self.Num_wm),
            nn.ReLU(inplace = True),
            nn.Upsample(scale_factor=4, mode = 'bicubic')
            )
        self.img_tail = nn.Sequential(
            nn.Conv2d(self.Num_wm, self.Num_wm, 3, padding = 1),
            nn.BatchNorm2d(self.Num_wm),
            nn.ReLU(inplace = True),
            nn.Conv2d(self.Num_wm, 3, 1, padding =0))

        self.relu = torch.nn.ReLU(inplace=True)
        self.merge = nn.Conv2d(6,3,1,padding = 0)

    def forward(self, img, wm):

        wm = wm.reshape(wm.size(0),-1) # -1 * 1 * H_wm * W_wm
        wm =wm.unsqueeze(-1).unsqueeze(-1) # -1 * Num_wm * 1 * 1

        img_r = self.img_resnet(img)   # -1 * Num_wm * 32 * 32
        img_r = self.img_expend(img_r) # -1 * Num_wm * 128 * 128

        img_wm = img_r * wm
        img_wm = self.relu(img_wm)
        img_wm = self.img_tail(img_wm)

        img_wm = torch.cat([img_wm, img],dim = 1)
        img_wm = self.merge(img_wm)

        return img_wm

