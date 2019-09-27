import torch
import torch.nn as nn
from lib.config import cfg

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res


class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class RCAN(nn.Module):
    def __init__(self, conv=default_conv):
        super(RCAN, self).__init__()
        
        


        self.H_img = cfg.DATA_SET.H_IMG 
        self.W_img = cfg.DATA_SET.W_IMG
        self.H_wm = cfg.DATA_SET.H_WM
        self.W_wm = cfg.DATA_SET.W_WM
        self.Num_wm = self.H_wm * self.W_wm 
        self.batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU
        

        
        
        n_resgroups = 5 # args.n_resgroups # 10 5
        n_resblocks = 16 # args.n_resblocks # 16 16
        n_feats = 64 # args.n_feats # 64
        kernel_size = 3
        reduction =  16 # args.reduction # 16 
        act = nn.ReLU(True)
        
        
        rgb_mean = (0.485, 0.456, 0.406) 
        rgb_std = (0.229, 0.224, 0.225) 
        
        self.sub_mean = MeanShift(255, rgb_mean, rgb_std) # args.rgb_range 255
        
        # define head module
        modules_head = [conv(3, n_feats, kernel_size)] # n_colors 3

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale= 1 , n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            nn.Conv2d(n_feats + self.Num_wm + 3, n_feats, kernel_size = 3, stride = 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, 3, kernel_size=1)
        ]

        self.add_mean = MeanShift(255, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)



    def forward(self, img, wm):
                
        wm = wm.reshape(self.batch_size,-1)
        wm =wm.unsqueeze(-1).unsqueeze(-1)
        wm = wm.expand(-1,-1, self.H_img, self.W_img)
        
        # x = self.sub_mean(img)
        x = self.head(img)

        res = self.body(x)
        res += x
        
        img_wm = torch.cat([ img,wm, res], dim=1)

        
        img_wm = self.tail(img_wm)
        # img_wm = self.add_mean(img_wm)

        return img_wm
