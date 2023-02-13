from basic_layers.lbr import LBR
import jax.numpy as jnp
from flax import linen as nn
class Encoder_base(nn.Module):
    data_shape:None
    batch_size:None
    latent_dim:None
    pca:None
    drop_prob:None
    hidden_dim: None

    def setup(self):
        self.fc_interior_1 = LBR(int(jnp.prod(self.data_shape)), self.hidden_dim,self.drop_prob)
        self.fc_interior_2 = LBR(self.hidden_dim, self.hidden_dim,self.drop_prob)
        self.fc_interior_3 = LBR(self.hidden_dim, self.hidden_dim,self.drop_prob)
        self.fc_interior_4 = LBR(self.hidden_dim, self.hidden_dim,self.drop_prob)
        self.fc_interior_5 = LBR(self.hidden_dim, self.hidden_dim,self.drop_prob)
        self.fc_interior_6 = LBR(self.hidden_dim, self.hidden_dim,self.drop_prob)
        self.fc_interior_7 = nn.Dense(self.latent_dim)
        self.batch_mu_1=nn.BatchNorm(use_scale=False,use_bias=False,use_running_average=False)


    def __call__(self, x):
        x=x.reshape(self.batch_size,-1)
        tmp=self.pca.transform(x)
        mu_1=self.fc_interior_7(self.fc_interior_6(self.fc_interior_5(self.fc_interior_4(self.fc_interior_3(self.fc_interior_2(self.fc_interior_1(tmp)))))))
        mu_1=self.batch_mu_1(mu_1)
        return mu_1
 
