import torch
import torch.nn as nn
import torch.nn.functional as F

class Horizontal(nn.Module):
    '''Sequential convolutional layers. The kernel size is fixed to 3 with padding 1
    to keep the input and output resolution the same. The number of channels/filters
    in each layer is specified by a list in the channels variable. There is also an
    option to add dropout to the layers.
    
    Attributes
    ----------
    
    length : int
        Number of convolutional layers in the module
    layers : nn.ModuleList
        A list of the sequential layers in the module, to be used during a call to forward
    '''
    
    def __init__(self, channels, dropout=None, last_layer=False):
        '''
        Parameters
        ----------
        
        channels : list(int)
            The number of channels in each tensor input/output tensor. The first element
            is the number of channels of the input tensor, and every other integer is the
            number of output channels of a convolutional layer. For example, the list
            [1, 2, 3] will create two convolutional layers which transforms a 1xWxH tensor
            into a 2xWxH tensor which transforms into a 3xWxH tensor.
        dropout : float, None, optional
            If None, no dropout layers are added. Otherwise, a dropout layer with probability
            dropout is after each convolutional layer. (default = None)
        last_layer : boolean, optional
            If True, no activation or dropout layer is added to the last convolutional layer
            in this module. This is intended to be used when this specific Horizontal instance
            is the last module in the entire network, and the output is meant to be the
            predicted image. In this case, no activation or dropout layers are added pixel
            values are intended to be in a standard normal distrubition. (default = False)
        
        '''
        super().__init__()
        self.length = len(channels)-1
        
        self.layers = nn.ModuleList()
        for l in range(self.length):
            in_channels = channels[l]
            out_channels = channels[l+1]
            
            self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            
            if not last_layer or l != self.length-1:
                self.layers.append(nn.BatchNorm2d(out_channels))
                self.layers.append(nn.ReLU())
                if dropout is not None:
                    self.layers.append(nn.Dropout2d(p=dropout))
        
    def forward(self, x):
        '''
        Transform the input tensor x through this module's layers
        
        Parameters
        ----------
        
        x : torch.tensor
            The input tensor
            
        Returns
        -------
        
        x : torch.Tensor
            The transformed tensor
        '''
        for l in self.layers:
            x = l(x)
        return x

class Down(nn.Module):
    '''A module to halve the image size followed by a Horizontal module.
    The image size is halved using a 2x2 MaxPooling layer.
    
    Attributes
    ----------
    
    layers : nn.Sequential
        The combined module layers
    '''
    
    def __init__(self, channels, dropout=None):
        '''
        Parameters
        ----------
        
        channels : list(int)
            A list of number of channels to be passed to the Horizontal module
        dropout : float, None, optional
            A dropout parameter to be passed to Horizontal (default = None)
        '''
    
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.MaxPool2d(2),
            Horizontal(channels, dropout=dropout)
        )
    
    def forward(self, x):
        '''
        Transform the input tensor x through this module's layers
        
        Parameters
        ----------
        
        x : torch.tensor
            The input tensor
        
        Returns
        -------
        
        x : torch.Tensor
            The transformed tensor
        '''
        return self.layers(x)
    
class Up(nn.Module):
    '''A module to double the image size followed by a horizontal module. Upscaling is
    done through either a transposed convolutional layer or through bilinear upsampling.
    Following the upscaling layer, the tensor is concatenated with an equivalently sized
    tensor from a preceeding Down module. This tensor is fed into forward.
    
    ####### NOTE ########
    The bilinear mode is currently untested, only the transposed convolutional mode has
    been tested
    
    Attributes
    ----------
    
    up : nn.ConvTranspose2d, nn.UpSample
        The upscaling layer, can be either a transposed convolutional layer or upsampling
        layer
    horiz : Horizontal
        The horizontal layer module
    '''
    
    def __init__(self, channels, dropout=None, bilinear=False):
        '''
        Parameters
        ----------
        
        channels : list(int)
            A list of number of channels to be passed to the Horizontal module
        dropout : float, None, optional
            A dropout parameter to be passed to Horizontal (default = None)
        bilinear : boolean, optinal
            If True, uses bilinear upsampling. Otherwise, uses a transposed convolutional
            layer (default = False)
        
        '''
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        else:
            self.up = nn.ConvTranspose2d(channels[0], channels[0] // 2, stride=2, kernel_size=2)
        self.horiz = Horizontal(channels, dropout=dropout)
            
    def forward(self, from_down, from_across):
        '''
        Takes tensors from the layer beneath this Up module as well as the
        tensor from the Down layer from across the U-Net and concatenates these tensors
        then transforms them according to the layers in this module.
        
        Parameters
        ----------
        
        from_down : torch.tensor
            The input tensor from the bottom layers. This tensor is upscaled
            so as to have the same resolution as from_across
        from_across : torch.tensor
            The input tensor from across the U-Net, having the same image
            resolution as the upscaled from_down tensor. 
            
        Returns
        -------
        
        x : torch.Tensor
            The transformed tensor
        '''
        from_down = self.up(from_down)
        x = torch.cat([from_across, from_down], dim=1)
        return self.horiz(x)

        
class UNet(nn.Module):
    '''This module uses the Up, Down, and Horizontal modules to
    implement a U-Net. The number of channels follows from the architecture
    laid out in the paper: https://arxiv.org/pdf/1505.04597.pdf
    
    Attributes
    ----------
    in_horiz : Horizontal
        The first sequential layers of convolution
    down_i : Down
        The downscaling modules. The integer i represents the depth
        of this layer
    up_i : Up
        The upscaling modules. The integer i represnets the depth
        of this layer
    out_horiz :
        The final sequential layers of convolution, outputting the final
        predicted image
    
    '''
    
    def __init__(self, in_channels=1, out_channels=1, dropout=None, bilinear=False):
        '''
        Parameters
        ----------
        n_channels : int, optional
            The number of channels in the input image (default = 1)
        dropout : float, None, optional
            A dropout parameter to be passed to Horizontal (default = None)
        bilinear : boolean, optional
            If True, uses bilinear upsampling. Otherwise, uses a transposed convolutional
            layer (default = False)
        
        '''
        super().__init__()
        
        self.in_horiz = Horizontal([in_channels, 64, 64], dropout=dropout)
        self.down_1 = Down([64, 128, 128], dropout=dropout)
        self.down_2 = Down([128, 256, 256], dropout=dropout)
        self.down_3 = Down([256, 512, 512], dropout=dropout)
        self.down_4 = Down([512, 1024, 1024], dropout=dropout)
        self.up_4 = Up([1024, 512, 512], dropout=dropout, bilinear=bilinear)
        self.up_3 = Up([512, 256, 256], dropout=dropout, bilinear=bilinear)
        self.up_2 = Up([256, 128, 128], dropout=dropout, bilinear=bilinear)
        self.up_1 = Up([128, 64, 64], dropout=dropout, bilinear=bilinear)
        self.out_horiz = Horizontal([64, out_channels, out_channels], dropout=dropout, last_layer=True)
        
    def forward(self, x):
        '''
        Transform the input tensor x through the U-Net's layers
        
        Parameters
        ----------
        
        x : torch.tensor
            The input tensor
            
        Returns
        -------
        
        x : torch.Tensor
            The transformed tensor
        '''
        x1 = self.in_horiz(x)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x4 = self.down_3(x3)
        x5 = self.down_4(x4)
        x = self.up_4(x5, x4)
        x = self.up_3(x, x3)
        x = self.up_2(x, x2)
        x = self.up_1(x, x1)
        x = self.out_horiz(x)
        return x
        
        #Receptive field is 21 pixels ??????