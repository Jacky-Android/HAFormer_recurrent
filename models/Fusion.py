import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=dilation, dilation=dilation, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x



class cwFModule(nn.Module):
    def __init__(self, in_channels):
        super(cwFModule, self).__init__()
        self.conv3x3 = DepthwiseSeparableConv(in_channels*2, in_channels*2, kernel_size=3)
        
        self.conv1x1 = nn.Conv2d(in_channels*2, in_channels*2, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.convout = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)
        

    def forward(self, x1, x2):

        x = torch.concat([x1,x2],dim=1)
        x = self.conv3x3(x)
        x = self.conv1x1(x)
        x = self.gap(x)
        x = self.convout(x)
        x = nn.Sigmoid()(x)
        x = torch.mul(x1,x)+torch.mul(x2,x)
        x = nn.ReLU6()(x)
        

        return x

    

# Example usage
if __name__ == "__main__":
    input_tensor1 = torch.randn(1, 64, 128, 128)  # Example input 1
    input_tensor2 = torch.randn(1, 64, 128, 128)  # Example input 2
    model = cwFModule(in_channels=64)
    output = model(input_tensor1, input_tensor2)
    print(output.shape)
