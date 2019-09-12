
import torch
import torch.nn as nn
import torchvision.models as models



def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class DilationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample = None, dilation = 1):
        super(DilationBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                          kernel_size = 3, stride=1, padding=dilation, dilation=dilation)
        self.bn1 =  nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        
        self.conv2 =  nn.Conv2d(out_channels, out_channels,
                          kernel_size = 3, stride=1, padding=dilation, dilation=dilation)
        self.bn2 =  nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out
        

class my_resnet5_7(nn.Module):
    def __init__(self):
        super(my_resnet5_7, self).__init__()
        
        

        self.channels = 64
        self.layer5  = self._make_layer(DilationBlock, 128, 2)
        self.layer6  = self._make_layer(DilationBlock, 256, 2)
        self.layer7  = self._make_layer(DilationBlock, 512, 2)
                     

    def _make_layer(self, block, out_channels, blocks):
        
        downsample = nn.Sequential(
                conv1x1(self.channels, out_channels , stride = 1),
                nn.BatchNorm2d(out_channels),)
        layers = []
        layers.append(block(self.channels, out_channels, downsample, dilation = 2))
        self.channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self,x):
        out = self.layer5(x)
        out = self.layer6(out)
        out = self.layer7(out)
#         out = self.layer6(out)
        return out


resnet18_model=models.resnet18(pretrained=True)

    
class resnet18_dilation(nn.Module):
    def __init__(self, res_model = resnet18_model):
        super(resnet18_dilation, self).__init__()
        
        self.layers_0_4 = torch.nn.Sequential(*list(res_model.children())[:5])
        self.layers_5_7 = my_resnet5_7()
        
    def forward(self, x):
        out = self.layers_0_4(x)
        out = self.layers_5_7(out)
        return out




def main():
    model = resnet18_dilation()
    model=model.eval()

    input_img = torch.rand([2, 3, 64, 64])
    output_img = model(input_img)
    print('input size: ', input_img.size())
    print('output size: ', output_img.size())
    print()

    input_img = torch.rand([2, 3, 128, 128])
    output_img = model(input_img)
    print('input size: ', input_img.size())
    print('output size: ', output_img.size())
    print()





if __name__ == '__main__':
    main()

