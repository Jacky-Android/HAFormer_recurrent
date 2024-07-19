# HAFormer_recurrent

在作者发布之前尝试复现，坐等作者发布🤓🤓🤓[HAFormer](https://github.com/XU-GITHUB-curry/HAFormer)

Paper: HAFormer: Unleashing the Power of  Hierarchy-Aware Features for Lightweight  Semantic Segmentation

[1] G. Xu, W. Jia, T. Wu, L. Chen, and G. Gao, “HAFormer: Unleashing the Power of Hierarchy-Aware Features for Lightweight Semantic Segmentation,” IEEE Trans. on Image Process., pp. 1–1, 2024, doi: 10.1109/TIP.2024.3425048.
# 模型结构
```python
==============================================================================================================
Layer (type:depth-idx)                                       Output Shape              Param #
==============================================================================================================
HAPEFormer                                                   [1, 6, 1024, 1024]        147,648
├─FeatureListNet: 1-1                                        [1, 48, 256, 256]         --
│    └─ConvNormAct: 2-1                                      [1, 24, 512, 512]         --
│    │    └─Dropout: 3-1                                     [1, 3, 1024, 1024]        --
│    │    └─Conv2d: 3-2                                      [1, 24, 512, 512]         648
│    │    └─BatchNorm2d: 3-3                                 [1, 24, 512, 512]         48
│    │    └─Hardswish: 3-4                                   [1, 24, 512, 512]         --
│    └─ResidualBlock: 2-2                                    [1, 24, 512, 512]         --
│    │    └─Identity: 3-5                                    [1, 24, 512, 512]         --
│    │    └─DSConv: 3-6                                      [1, 24, 512, 512]         888
│    │    └─Identity: 3-7                                    [1, 24, 512, 512]         --
│    └─EfficientVitStage: 2-3                                [1, 48, 256, 256]         --
│    │    └─Sequential: 3-8                                  [1, 48, 256, 256]         50,304
│    └─EfficientVitStage: 2-4                                [1, 96, 128, 128]         --
│    │    └─Sequential: 3-9                                  [1, 96, 128, 128]         267,072
│    └─EfficientVitStage: 2-5                                [1, 192, 64, 64]          --
│    │    └─Sequential: 3-10                                 [1, 192, 64, 64]          2,200,320
│    └─EfficientVitStage: 2-6                                [1, 384, 32, 32]          --
│    │    └─Sequential: 3-11                                 [1, 384, 32, 32]          12,457,728
├─Downsample: 1-2                                            [1, 32, 512, 512]         --
│    └─Conv2d: 2-7                                           [1, 32, 512, 512]         896
├─SeparableConvBN: 1-3                                       [1, 32, 512, 512]         --
│    └─Conv2d: 2-8                                           [1, 32, 512, 512]         288
│    └─BatchNorm2d: 2-9                                      [1, 32, 512, 512]         64
│    └─Conv2d: 2-10                                          [1, 32, 512, 512]         1,024
├─Downsample: 1-4                                            [1, 32, 256, 256]         --
│    └─Conv2d: 2-11                                          [1, 32, 256, 256]         9,248
├─ModuleList: 1-13                                           --                        (recursive)
│    └─Sequential: 2-12                                      [1, 32, 256, 256]         --
│    │    └─HAPEBlock: 3-12                                  [1, 32, 256, 256]         3,856
│    │    └─HAPEBlock: 3-13                                  [1, 32, 256, 256]         3,856
│    │    └─HAPEBlock: 3-14                                  [1, 32, 256, 256]         3,856
├─Downsample: 1-6                                            [1, 64, 128, 128]         --
│    └─Conv2d: 2-13                                          [1, 64, 128, 128]         18,496
├─ModuleList: 1-13                                           --                        (recursive)
│    └─Sequential: 2-14                                      [1, 64, 128, 128]         --
│    │    └─HAPEBlock: 3-15                                  [1, 64, 128, 128]         6,992
│    │    └─HAPEBlock: 3-16                                  [1, 64, 128, 128]         6,992
│    │    └─HAPEBlock: 3-17                                  [1, 64, 128, 128]         3,912
│    │    └─HAPEBlock: 3-18                                  [1, 64, 128, 128]         3,912
│    │    └─HAPEBlock: 3-19                                  [1, 64, 128, 128]         2,372
│    │    └─HAPEBlock: 3-20                                  [1, 64, 128, 128]         2,372
├─Conv2d: 1-8                                                [1, 128, 128, 128]        8,192
├─ModuleList: 1-13                                           --                        (recursive)
│    └─Sequential: 2-15                                      [1, 128, 128, 128]        --
│    │    └─HAPEBlock: 3-21                                  [1, 128, 128, 128]        26,272
│    │    └─HAPEBlock: 3-22                                  [1, 128, 128, 128]        26,272
│    │    └─HAPEBlock: 3-23                                  [1, 128, 128, 128]        13,968
│    │    └─HAPEBlock: 3-24                                  [1, 128, 128, 128]        13,968
│    │    └─HAPEBlock: 3-25                                  [1, 128, 128, 128]        7,816
│    │    └─HAPEBlock: 3-26                                  [1, 128, 128, 128]        7,816
├─Conv2d: 1-10                                               [1, 8192, 32, 32]         3,145,728
├─cwFModule: 1-11                                            [1, 128, 256, 256]        --
│    └─DepthwiseSeparableConv: 2-16                          [1, 256, 256, 256]        --
│    │    └─Conv2d: 3-27                                     [1, 256, 256, 256]        2,560
│    │    └─Conv2d: 3-28                                     [1, 256, 256, 256]        65,792
│    └─Conv2d: 2-17                                          [1, 256, 256, 256]        65,792
│    └─AdaptiveAvgPool2d: 2-18                               [1, 256, 1, 1]            --
│    └─Conv2d: 2-19                                          [1, 128, 1, 1]            32,896
├─Conv2d: 1-12                                               [1, 64, 256, 256]         8,192
├─ModuleList: 1-13                                           --                        (recursive)
│    └─Sequential: 2-20                                      [1, 64, 256, 256]         --
│    │    └─HAPEBlock: 3-29                                  [1, 64, 256, 256]         14,880
│    │    └─HAPEBlock: 3-30                                  [1, 64, 256, 256]         14,880
│    │    └─HAPEBlock: 3-31                                  [1, 64, 256, 256]         14,880
├─Conv2d: 1-14                                               [1, 6, 512, 512]          384
==============================================================================================================
```

# 模型FloPs
```python
==============================================================================================================
| module                           | #parameters or shape   | #flops     |
|:---------------------------------|:-----------------------|:-----------|
| model                            | 7.281M                 | 28.863G    |
==============================================================================================================
```
## HAPE Blocks
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PEM(nn.Module):
    def __init__(self, in_channels):
        super(PEM, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.gap(x)
        x1 = x1.view(x1.size(0), -1)
        A = F.softmax(x1, dim=1).view(x1.size(0), x1.size(1), 1, 1)
        x = x * A + x
        return x

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # Reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # Transpose
    x = torch.transpose(x, 1, 2).contiguous()

    # Flatten
    x = x.view(batchsize, -1, height, width)

    return x
   
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
class HAPEBlock(nn.Module):
    def __init__(self, in_channels, denote=4):
        super(HAPEBlock, self).__init__()
        com_channels = in_channels // denote
        self.d = denote
        self.conv_in = nn.Conv2d(in_channels, com_channels, kernel_size=1)

        self.convs = nn.ModuleList()
        self.PEMs = nn.ModuleList()
        
        # Define convolutional layers based on the denote value
        for i in range(1, denote+1):
            if i%4 == 1:
                self.convs.append(nn.Sequential(
                    nn.Conv2d(com_channels, com_channels, kernel_size=(1, 3), padding=(0, 1)),
                    nn.Conv2d(com_channels, com_channels, kernel_size=(3, 1), padding=(1, 0))
                ))
            elif i%4 == 2:
                self.convs.append(nn.Sequential(
                    DepthwiseSeparableConv(com_channels, com_channels, kernel_size=(3, 1), padding=(1, 0)),
                    DepthwiseSeparableConv(com_channels, com_channels, kernel_size=(1, 3), padding=(0, 1))
                ))
            elif i%4 == 3:
                self.convs.append(nn.Sequential(
                    DepthwiseSeparableConv(com_channels, com_channels, kernel_size=(5, 1), padding=(2, 0)),
                    DepthwiseSeparableConv(com_channels, com_channels, kernel_size=(1, 5), padding=(0, 2))
                ))
            elif i%4 == 0:
                self.convs.append(nn.Sequential(
                    DepthwiseSeparableConv(com_channels, com_channels, kernel_size=(7, 1), padding=(3, 0)),
                    DepthwiseSeparableConv(com_channels, com_channels, kernel_size=(1, 7), padding=(0, 3))
                ))
            self.PEMs.append(PEM(in_channels=com_channels))
        
        self.conv_out = nn.Conv2d(com_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        shot = x
        x_in = self.conv_in(x)
        
        # Process each convolution and PEM
        outputs = []
        for i in range(self.d):
            x_conv = self.convs[i](x_in)
            
            x_pem = self.PEMs[i](x_conv)
            outputs.append(x_pem)
        
        x = sum(outputs)
        x = self.conv_out(x) + shot
        x = nn.ReLU6()(x)
        x = channel_shuffle(x, self.d)
        return x

# Example usage
if __name__ == "__main__":
    input_tensor = torch.randn(1, 64, 64, 64)  # Example input
    model = HAPEBlock(in_channels=64, denote=16)  # Change denote to 6
    output = model(input_tensor)
    print(output.shape)
############################
torch.Size([1, 64, 64, 64])
############################
```


