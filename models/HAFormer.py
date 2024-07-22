import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .HAPE import HAPEBlock
from .Fusion import cwFModule
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis, flop_count_table

class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )  
            
    
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, method='conv', kernel_size=3, stride=2, padding=1):
        super(Downsample, self).__init__()
        if method == 'conv':
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        elif method == 'maxpool':
            self.downsample = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        elif method == 'avgpool':
            self.downsample = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            raise ValueError("Unsupported downsample method: {}".format(method))
        
    def forward(self, x):
        return self.downsample(x)

class HAPEFormer(nn.Module):
    def __init__(self, vit = 'efficientvit_b3',channels=[32, 64, 128, 64], depths=[[2, 2, 2], [4, 4, 8, 8, 16, 16], [4, 4, 8, 8, 16, 16], [2, 2, 2]], num_classes=6):
        super(HAPEFormer, self).__init__()
        self.vit = timm.create_model(vit, pretrained=True, num_classes=0, features_only=True)
        self.down0 = Downsample(3, out_channels=channels[0])
        self.cnn_stem = SeparableConvBN(in_channels=channels[0], out_channels=channels[0])
        
        self.stages = nn.ModuleList()
        for i, (channel, depth) in enumerate(zip(channels, depths)):
            stage = nn.Sequential()
            for j in range(len(depth)):
                stage.add_module(f"HAPEBlock_{i}_{j}", HAPEBlock(in_channels=channel, denote=depth[j]))
            self.stages.append(stage)
        
        self.cnn = nn.Conv2d(channels[1], channels[2], kernel_size=1, bias=False)

        self.down1 = Downsample(in_channels=channels[0], out_channels=channels[0])
        self.down2 = Downsample(in_channels=channels[0], out_channels=channels[1])
        self.down3 = Downsample(in_channels=channels[1], out_channels=channels[2])
        self.down4 = Downsample(in_channels=channels[2], out_channels=channels[3])

        encoder_channels = self.vit.feature_info.channels()[-2]
        outencoder_channels = int(((channels[2] / encoder_channels) * 16) * encoder_channels)
        
        self.cnn1 = nn.Conv2d(encoder_channels, outencoder_channels, kernel_size=1, bias=False)
        self.fusion = cwFModule(in_channels=channels[2])

        self.cnn2 = nn.Conv2d(in_channels=channels[2], out_channels=channels[3], kernel_size=1, bias=False)

        self.classer = nn.Conv2d(in_channels=channels[3], out_channels=num_classes, kernel_size=1, bias=False)
        
                
    def forward(self, x):
        vit = self.vit(x)[-2]
        x = self.down0(x)
        x = self.cnn_stem(x)

        x = self.down1(x)
        x = self.stages[0](x)

        x = self.down2(x)
        x = self.stages[1](x)
        
        x = self.cnn(x)
        x = self.stages[2](x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        b, c, h, w = x.shape
        #print(x.shape,vit.shape)
        vit = self.cnn1(vit)
        vit = vit.view(b, c, h, w)
        x = self.fusion(x, vit)
        x = self.stages[3](self.cnn2(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.classer(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return x

# Example usage

if __name__ == "__main__":
    input_tensor = torch.randn(1, 3, 512, 512)  # Example input
    model = HAPEFormer()  # Change denote to 6
    output = model(input_tensor)
    print(output.shape)
    summary(model,[1,3,512,512])
    