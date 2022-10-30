# Model Building
import torch
from torch import nn, optim
import torch.nn.functional as F

# Import Generator, Discriminator
from discriminator import Discriminator
from generator import Generator

# Model maintenance
import wandb

class ReagingGAN(nn.Module):
    def __init__(self, input_shape: int = 128) -> None:
        super().__init__()
        self.generator = Discriminator(input_size=input_shape)
        self.discriminator = Generator(input_size=input_shape)
        
        
