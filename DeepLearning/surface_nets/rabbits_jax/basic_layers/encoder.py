from basic_layers.lbr import LBR
import jax.numpy as jnp
from flax import linen as nn
class Encoder_base(nn.Module):
    latent_dim:None
    hidden_dim: None
    pca: None
    
    @nn.compact
    def __call__(self, x, training):
        x=self.pca.transform(x)
        x=LBR(self.hidden_dim)(x, training)
        x=LBR(self.hidden_dim)(x, training)
        x=LBR(self.hidden_dim)(x, training)
        x=LBR(self.hidden_dim)(x, training)
        x=LBR(self.hidden_dim)(x, training)
        x=LBR(self.hidden_dim)(x, training)
        x=LBR(self.hidden_dim)(x, training)
        x=LBR(self.hidden_dim)(x, training)
        mu=nn.Dense(self.latent_dim)(x)
        #mu=nn.BatchNorm(use_running_average=False)(mu)
        logsigma=nn.Dense(self.latent_dim)(x)
        return mu,logsigma
 
