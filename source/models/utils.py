import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int, 
                 stride: int, 
                 padding: int, 
                 use_bn: bool = True, 
                 use_drop: bool = False) -> None:
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
        if use_bn:
            layers.append(nn.BatchNorm2d(num_features=out_channels))
        layers.append(nn.ReLU())
        if use_drop:
            layers.append(nn.Dropout(0.3))
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Residual(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int, 
                 stride: int, 
                 padding: int) -> None:
        super().__init__()
        
    