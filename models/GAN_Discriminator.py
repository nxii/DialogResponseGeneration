import torch
import torch.nn as nn

class GAN_Discriminator(nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.discriminator = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size), nn.BatchNorm1d(self.hidden_size), nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_size, self.hidden_size), nn.BatchNorm1d(self.hidden_size), nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_size, 1)
        )
    
    def forward(self, x):
        return self.discriminator(x)
