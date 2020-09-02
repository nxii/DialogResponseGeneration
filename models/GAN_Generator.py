import torch
import torch.nn as nn

class GAN_Generator(nn.Module):
    
    def __init__(self, input_size, latent_size):
        super().__init__()
        
        self.input_size = input_size
        self.latent_size = latent_size
                
        self.generator = nn.Sequential(
            nn.Linear(self.input_size, 2*self.input_size), nn.BatchNorm1d(2*self.input_size), nn.LeakyReLU(0.2), nn.Dropout(p=0.33),
            nn.Linear(2*self.input_size, 2*self.input_size), nn.BatchNorm1d(2*self.input_size), nn.LeakyReLU(0.2), nn.Dropout(p=0.33),
            nn.Linear(2*self.input_size, self.latent_size)
        )
        
    def forward(self, x):
        return self.generator(x) 
