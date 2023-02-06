from torch import nn
from models.basic_layers.lbr import LBR
import torch

class Variance_estimator(nn.Module):
    def __init__(self,latent_dim,hidden_dim,data_shape,drop_prob):
        super().__init__()
        self.latent_dim=latent_dim
        self.var = LBR(latent_dim, latent_dim, drop_prob)
        self.batch_sigma=nn.BatchNorm1d(self.latent_dim)

    def forward(self,mu):
        sigma=self.batch_sigma(self.var(mu))
        sigma=torch.exp(sigma)
        return sigma
 
