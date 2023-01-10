from torch import nn
from models.basic_layers.lbr import LBR
import torch

class Variance_estimator(nn.Module):
    def __init__(self,latent_dim_1,latent_dim_2,hidden_dim,data_shape,drop_prob):
        super().__init__()
        self.latent_dim_1=latent_dim_1
        self.latent_dim_2=latent_dim_2
        self.var1 = LBR(latent_dim_1, latent_dim_1,drop_prob)
        self.var2 = LBR(latent_dim_2, latent_dim_2, drop_prob)
        self.batch_sigma1=nn.BatchNorm1d(self.latent_dim_1)
        self.batch_sigma2=nn.BatchNorm1d(self.latent_dim_2)

    
    def forward(self,mu_1,mu_2):
        sigma_1=self.batch_sigma1(self.var1(mu_1))
        sigma_1=torch.exp(sigma_1)
        sigma_2=self.batch_sigma2(self.var2(mu_2))
        sigma_2=torch.exp(sigma_2)
        return sigma_1,sigma_2
 
