from basic_layers.lbr import LBR
import jax.numpy as jnp
from flax import linen as nn
class Encoder_base(nn.Module):
    latent_dim:None
    hidden_dim: None
    
    @nn.compact
    def __call__(self, x):
        x=LBR(self.hidden_dim)(x)
        x=LBR(self.hidden_dim)(x)
        x=LBR(self.hidden_dim)(x)
        x=LBR(self.hidden_dim)(x)
        x=nn.Dense(self.latent_dim)(x)
        return x
 
