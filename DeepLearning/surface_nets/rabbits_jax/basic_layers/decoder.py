from basic_layers.lbr import LBR
import jax.numpy as jnp
from flax import linen as nn

class Decoder_base(nn.Module):
    size:None
    hidden_dim: None
    pca: None
    barycenter: None

    @nn.compact
    def __call__(self, x, training):
        x=LBR(self.hidden_dim)(x, training)
        x=LBR(self.hidden_dim)(x, training)
        x=LBR(self.hidden_dim)(x, training)
        x=LBR(self.hidden_dim)(x, training)
        x=LBR(self.hidden_dim)(x, training)
        x=LBR(self.hidden_dim)(x, training)
        x=LBR(self.hidden_dim)(x, training)
        x=LBR(self.hidden_dim)(x, training)
        x=nn.Dense(self.size)(x)
        x=self.pca.inverse_transform(x)
        x=x.reshape(x.shape[0],-1,3)
        x=x-jnp.expand_dims(jnp.mean(x,axis=1),axis=1).repeat(x.shape[1],axis=1)+jnp.expand_dims(jnp.expand_dims(self.barycenter,0),0).repeat(x.shape[1],axis=1).repeat(x.shape[0],axis=0)
        x=x.reshape(x.shape[0],-1)
        return x
 
