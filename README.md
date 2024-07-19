# HAFormer_recurrent

在作者发布之前尝试复现，坐等作者发布🤓🤓🤓[HAFormer](https://github.com/XU-GITHUB-curry/HAFormer)

Paper: HAFormer: Unleashing the Power of  Hierarchy-Aware Features for Lightweight  Semantic Segmentation

[1] G. Xu, W. Jia, T. Wu, L. Chen, and G. Gao, “HAFormer: Unleashing the Power of Hierarchy-Aware Features for Lightweight Semantic Segmentation,” IEEE Trans. on Image Process., pp. 1–1, 2024, doi: 10.1109/TIP.2024.3425048.
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


