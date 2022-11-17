import torch
import torch.nn as nn
import torch.nn.functional as F
from generator import Generator
from discriminator import Discriminator
from tqdm import tqdm

class ReAgingGAN:
    def __init__(self, 
                 discriminator: Discriminator, 
                 generator: Generator,
                 optimizers: dict, 
                 schedulers: dict, 
                 trainloader: torch.utils.data.DataLoader, 
                 validloader: torch.utils.data.DataLoader, 
                 config: dict) -> None:
        self.discriminator = discriminator
        self.generator = generator
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.bce = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        self.trainloader = trainloader
        self.validloader = validloader
        self.config = config
    
    def train(self):
        for epoch in range(self.config['EPOCHS']):
            for batch_ix, (X, y) in tqdm(enumerate(self.trainloader), desc=f'(TRAIN) EPOCH [{epoch+1}/{self.config["EPOCHS"]}]'):
                in_img, in_age = X[0].to(self.config['DEVICE']), X[1].to(self.config['DEVICE'])
                out_img, out_age = y[0].to(self.config['DEVICE']), y[1].to(self.config['DEVICE'])
                
                x_dash = self.generator(in_img, in_age, out_age)
                x_rec  = self.generator(in_img, in_age, in_age)
                x_cyc  = self.generator(x_dash, out_age, in_age)
                
                adv_loss = self._adversarial_loss(in_img, in_age, out_img, out_age, x_dash)
                rec_loss = self._reconstruction_loss(in_img, x_rec)
                cyc_loss = self._cycle_consistency_loss(in_img, x_cyc)
                
                disc_loss = adv_loss[0]
                gen_loss = adv_loss[1]
                
                gen_loss = self.config['ADV_LAMBDA'] * gen_loss + self.config['REC_LAMBDA'] * rec_loss + self.config['CYC_LAMBDA'] * cyc_loss
                
                self.optimizers['discriminator'].zero_grad()
                disc_loss.backward(retain_graph=True)
                self.optimizers['discriminator'].step()
                self.schedulers['discriminator'].step()
                
                self.optimizers['generator'].zero_grad()
                gen_loss.backward()
                self.optimizers['generator'].step()
                self.schedulers['generator'].step()
    
    def _adversarial_loss(self, in_img: torch.Tensor, in_age: torch.Tensor, out_img: torch.Tensor, out_age: torch.Tensor, x_dash: torch.Tensor) -> torch.Tensor:
        disc_fake = self.discriminator(in_img, in_age, x_dash, out_age)
        disc_real = self.discriminator(in_img, in_age, out_img, out_age)
        
        disc_real_loss = self.bce(disc_real, torch.ones_like(disc_real))
        disc_fake_loss = self.bce(disc_fake, torch.zeros_like(disc_fake))
        
        gen_loss = self.bce(disc_fake, torch.ones_like(disc_fake))
        
        return (disc_fake_loss + disc_real_loss) / 2, gen_loss
    
    def _reconstruction_loss(self, in_img: torch.Tensor, x_rec: torch.Tensor) -> torch.Tensor:
        return self.l1_loss(in_img, x_rec)
    
    def _cycle_consistency_loss(self, in_img: torch.Tensor, x_cyc: torch.Tensor) -> torch.Tensor:
        return self.l1_loss(in_img, x_cyc)