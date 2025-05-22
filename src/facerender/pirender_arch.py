import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.norm1 = nn.InstanceNorm2d(dim)
        self.norm2 = nn.InstanceNorm2d(dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual
        return x

class PIRenderArch(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9):
        super(PIRenderArch, self).__init__()
        
        # Initial convolution
        model = [nn.Conv2d(input_nc, ngf, 7, 1, 3),
                nn.InstanceNorm2d(ngf),
                nn.ReLU(inplace=True)]
        
        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1),
                     nn.InstanceNorm2d(ngf * mult * 2),
                     nn.ReLU(inplace=True)]
        
        # Residual blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResBlock(ngf * mult)]
        
        # Upsampling
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), 3, stride=2, padding=1, output_padding=1),
                     nn.InstanceNorm2d(int(ngf * mult / 2)),
                     nn.ReLU(inplace=True)]
        
        # Output convolution
        model += [nn.Conv2d(ngf, output_nc, 7, 1, 3),
                 nn.Tanh()]
        
        self.model = nn.Sequential(*model)
        
        # Coefficient processing
        self.coeff_net = nn.Sequential(
            nn.Linear(80, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True)
        )
        
        # Fusion layer
        self.fusion = nn.Conv2d(ngf + 64, ngf, 1)
    
    def forward(self, source, coeff):
        # Process coefficients
        coeff = coeff.view(coeff.size(0), -1)
        coeff_feat = self.coeff_net(coeff)
        coeff_feat = coeff_feat.view(coeff_feat.size(0), 64, 4, 4)
        coeff_feat = F.interpolate(coeff_feat, size=source.size()[2:], mode='bilinear', align_corners=False)
        
        # Process source image
        feat = self.model[:-2](source)  # Get features before last conv
        
        # Fuse features
        feat = torch.cat([feat, coeff_feat], dim=1)
        feat = self.fusion(feat)
        
        # Generate output
        out = self.model[-2:](feat)  # Apply last conv and tanh
        
        return out 
