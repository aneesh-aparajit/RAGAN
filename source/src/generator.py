import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2) -> None:
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=stride, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TransposeConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2) -> None:
        super(TransposeConvBlock, self).__init__()
        self.tran_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, 
                               stride=stride, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tran_conv(x)


class Generator(nn.Module):
    """This is essentially a U-Net type architecture. But, we'll pass the input age and the output age as well.
    
    So, if the in_channels = 3, then we'll pass 5 channels in the model.
        - 1 will be for the input_age embedding.
        - 2 will be for the output_age embedding.
    
    """

    def __init__(self, in_channels: int = 3, num_age_groups: int = 3, input_size: int = 128) -> None:
        super(Generator, self).__init__()
        
        self.input_embed = nn.Embedding(num_embeddings=num_age_groups, embedding_dim=input_size*input_size)
        self.output_embed = nn.Embedding(num_embeddings=num_age_groups, embedding_dim=input_size*input_size)
        self.input_size = input_size


    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


