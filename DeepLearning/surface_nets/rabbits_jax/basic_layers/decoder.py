from basic_layers.lbr import LBR
from basic_layers.barycentre import Barycentre

import numpy as np
from flax import linen as nn

class Decoder_base(nn.Module):
    size:None
    hidden_dim: None

    @nn.compact
    def __call__(self, x):
        x=LBR(self.hidden_dim)(x)
        x=LBR(self.hidden_dim)(x)
        x=LBR(self.hidden_dim)(x)
        x=LBR(self.hidden_dim)(x)
        x=nn.Dense(self.size)(x)

        return x
 
