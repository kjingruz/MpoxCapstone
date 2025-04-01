import torch
import torch.nn as nn

class UNetEncoder(nn.Module):
    """
    Extract encoder part from UNet for self-supervised learning and transfer learning
    """
    def __init__(self, unet_model):
        super(UNetEncoder, self).__init__()
        
        # Extract the encoder part from UNet
        # This depends on the specific UNet implementation
        self.inc = unet_model.inc
        self.down1 = unet_model.down1
        self.down2 = unet_model.down2
        self.down3 = unet_model.down3
        self.down4 = unet_model.down4
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x5  # Return the bottleneck features
