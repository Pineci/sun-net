import torch
import torch.nn as nn
import torch.nn.functional as F

class Pixelwise(nn.Module):
    
    def __init__(self, n_channels=1):
        super().__init__()
        
        self.conv1_1 = nn.Conv2d(n_channels, 64, kernel_size=1, padding=0)
        self.bano1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU()
        
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=1, padding=0)
        self.bano2_1 = nn.BatchNorm2d(128)
        self.relu2_1 = nn.ReLU()
        
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=1, padding=0)
        self.bano3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU()
        
        self.last_1 = nn.Conv2d(256, 64, kernel_size=1, padding=0)
        self.last_2 = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        
        
    def forward(self, x):
        x = self.relu1_1(self.bano1_1(self.conv1_1(x)))
        x = self.relu2_1(self.bano2_1(self.conv2_1(x)))
        x = self.relu3_1(self.bano3_1(self.conv3_1(x)))
        
        x = self.last_2(self.last_1(x))
        return x