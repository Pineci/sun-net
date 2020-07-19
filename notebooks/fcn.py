import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    
    def __init__(self, mode, n_channels=1, single_pixel_pred=False):
        self.mode = mode
        super().__init__()
        
        if single_pixel_pred:
            ks, pd = 1, 0
        else:
            ks, pd = 3, 1
        
        #conv1
        self.conv1_1 = nn.Conv2d(n_channels, 64, kernel_size=ks, padding=pd)
        self.bano1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=ks, padding=pd)
        self.bano1_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, stride=2)
        
        #conv2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=ks, padding=pd)
        self.bano2_1 = nn.BatchNorm2d(128)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=ks, padding=pd)
        self.bano2_2 = nn.BatchNorm2d(128)
        self.relu2_2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, stride=2)
        
        #conv3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=ks, padding=pd)
        self.bano3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=ks, padding=pd)
        self.bano3_2 = nn.BatchNorm2d(256)
        self.relu3_2 = nn.ReLU()
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=ks, padding=pd)
        self.bano3_3 = nn.BatchNorm2d(256)
        self.relu3_3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, stride=2)
        
        #conv4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=ks, padding=pd)
        self.bano4_1 = nn.BatchNorm2d(512)
        self.relu4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=ks, padding=pd)
        self.bano4_2 = nn.BatchNorm2d(512)
        self.relu4_2 = nn.ReLU()
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=ks, padding=pd)
        self.bano4_3 = nn.BatchNorm2d(512)
        self.relu4_3 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, stride=2)
        
        #conv5
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=ks, padding=pd)
        self.bano5_1 = nn.BatchNorm2d(512)
        self.relu5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=ks, padding=pd)
        self.bano5_2 = nn.BatchNorm2d(512)
        self.relu5_2 = nn.ReLU()
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=ks, padding=pd)
        self.bano5_3 = nn.BatchNorm2d(512)
        self.relu5_3 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(2, stride=2)
        
        #conv6
        self.conv6 = nn.Conv2d(512, 4096, kernel_size=ks, padding=pd)
        self.bano6 = nn.BatchNorm2d(4096)
        self.relu6 = nn.ReLU()
        
        #conv7
        self.conv7 = nn.Conv2d(4096, 4096, kernel_size=ks, padding=pd)
        self.bano7 = nn.BatchNorm2d(4096)
        self.relu7 = nn.ReLU()
        
        self.last = nn.Conv2d(4096, 64, kernel_size=1)
        
        if self.mode == '32':
            self.upscale_conv_32 = nn.ConvTranspose2d(64, 1, kernel_size=32, stride=32)
        elif self.mode == '16':
            self.last_pool4 = nn.Conv2d(512, 64, 1)
            self.upscale_conv_2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
            self.upscale_conv_16 = nn.ConvTranspose2d(64, 1, kernel_size=16, stride=16)
        elif self.mode == '8':
            self.last_pool3 = nn.Conv2d(256, 64, 1)
            self.last_pool4 = nn.Conv2d(512, 64, 1)
            self.upscale_conv_4 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=4)
            self.upscale_conv_8 = nn.ConvTranspose2d(64, 1, kernel_size=8, stride=8)
            self.upscale_pool_2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        
        
    def forward(self, x):
        x = self.relu1_1(self.bano1_1(self.conv1_1(x)))
        x = self.relu1_2(self.bano1_2(self.conv1_2(x)))
        x = self.pool1(x)
        
        x = self.relu2_1(self.bano2_1(self.conv2_1(x)))
        x = self.relu2_2(self.bano2_2(self.conv2_2(x)))
        x = self.pool2(x)
        
        x = self.relu3_1(self.bano3_1(self.conv3_1(x)))
        x = self.relu3_2(self.bano3_2(self.conv3_2(x)))
        x = self.relu3_3(self.bano3_3(self.conv3_3(x)))
        x = self.pool3(x)
        if self.mode == '8':
            pool3 = x
        
        x = self.relu4_1(self.bano4_1(self.conv4_1(x)))
        x = self.relu4_2(self.bano4_2(self.conv4_2(x)))
        x = self.relu4_3(self.bano4_3(self.conv4_3(x)))
        x = self.pool4(x)
        if self.mode == '8' or self.mode == '16':
            pool4 = x
        
        x = self.relu5_1(self.bano5_1(self.conv5_1(x)))
        x = self.relu5_2(self.bano5_2(self.conv5_2(x)))
        x = self.relu5_3(self.bano5_3(self.conv5_3(x)))
        x = self.pool5(x)
        
        x = self.relu6(self.bano6(self.conv6(x)))
        x = self.relu7(self.bano7(self.conv7(x)))
        
        if self.mode == '32':
            x = self.upscale_conv_32(self.last(x))
        elif self.mode == '16':
            x = self.upscale_conv_2(self.last(x))
            x = self.upscale_conv_16(self.last_pool4(pool4) + x)
        elif self.mode == '8':
            #print(x.shape)
            #print(pool4.shape)
            x = self.last(x)
            #print(x.shape)
            x = self.upscale_conv_4(x)
            upscaled_pool4 = self.upscale_pool_2(self.last_pool4(pool4))
            #print(x.shape)
            #print(upscaled_pool4.shape)
            x = self.upscale_conv_8(self.last_pool3(pool3) + upscaled_pool4 + x)
        return x
    
    
    #Receptive field is 33?????