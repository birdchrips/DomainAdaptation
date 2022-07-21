import torch
from torch import nn
import matplotlib.pyplot as plt

class Discriminator(nn.Module):
    
    def __init__(self, n_hidden, LearningRate=0.01):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, 2)
        )

    
    def forward(self, inputs):
  
        return self.model(inputs)
   
        