import torch
import torch.nn as nn
from torch.nn.functional import log_softmax

class BOW_VAE(nn.Module):
    
    def __init__(self, hidden_size, latent_size, vocab_size, temperature):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.vocab_size = vocab_size
        self.temperature = temperature
        
        self.encoder = nn.Sequential(
                nn.Linear(self.vocab_size, 2*self.hidden_size), nn.BatchNorm1d(2*self.hidden_size), nn.LeakyReLU(0.2),
                nn.Linear(2*self.hidden_size, 2*self.hidden_size), nn.BatchNorm1d(2*self.hidden_size), nn.LeakyReLU(0.2),
                nn.Linear(2*self.hidden_size, self.hidden_size), nn.BatchNorm1d(self.hidden_size), nn.LeakyReLU(0.2)
            )
        
        self.hidden2mean = nn.Linear(self.hidden_size, self.latent_size)
        self.hidden2logv = nn.Linear(self.hidden_size, self.latent_size)
        
        #self.decoder = nn.Linear(self.latent_size, self.vocab_size)
        self.decoder = nn.Sequential(
                nn.Linear(self.latent_size, 2*self.hidden_size), nn.BatchNorm1d(2*self.hidden_size), nn.LeakyReLU(0.2),
                nn.Linear(2*self.hidden_size, self.vocab_size), nn.BatchNorm1d(vocab_size), nn.LeakyReLU(0.2)
            )
        
    
    def encode(self, batch_x):
        hidden = self.encoder(batch_x)
        
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        
        # Reparameterization trick
        eps = torch.randn(mean.size()).cuda()
        z = mean + (self.temperature * (eps * logv.exp()))
        
        return mean, logv, z
    
    
    def decode(self, z):
        logits = log_softmax(self.decoder(z), dim=-1)
        #logits = log_softmax(self.decoder(hidden), dim=-1)
        
        return logits
        
    
    def forward(self, batch_x):
        
        mean, logv, z = self.encode(batch_x)
        
        logits = self.decode(z)
        
        return mean, logv, z, logits
        #return hidden, logits 
