import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, download=True):
        super().__init__()
        if download:
            self.fcn = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
        self.fcn.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
    
    def forward(self, x):
        return self.fcn(x)       
