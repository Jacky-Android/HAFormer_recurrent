from torchinfo import summary
import torch
from models.HAFormer import HAPEFormer
from fvcore.nn import FlopCountAnalysis, flop_count_table
def calculate_flops(model, input_tensor):
    flops = FlopCountAnalysis(model, input_tensor)
    print(flop_count_table(flops))
input_tensor = torch.randn(1, 3, 512,512).cuda()  # Example input
model = HAPEFormer(vit = 'efficientvit_b1').cuda()  # Change denote to 6
output = model(input_tensor)

summary(model,[1,3,1024,1024])
calculate_flops(model, input_tensor)
