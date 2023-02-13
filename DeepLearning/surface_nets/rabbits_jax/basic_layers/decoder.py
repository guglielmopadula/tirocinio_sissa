from basic_layers.lbr import LBR
from basic_layers.barycentre import Barycentre

import numpy as np
from flax import linen as nn

class Decoder_base(nn.Module):
    data_shape:None
    batch_size:None
    latent_dim:None
    pca:None
    drop_prob:None
    hidden_dim: None
    barycenter:None
    def setup(self):
        self.fc_interior_1 = LBR(self.latent_dim, self.hidden_dim,self.drop_prob)
        self.fc_interior_2 = LBR(self.hidden_dim, self.hidden_dim,self.drop_prob)
        self.fc_interior_3 = LBR(self.hidden_dim, self.hidden_dim,self.drop_prob)
        self.fc_interior_4 = LBR(self.hidden_dim, self.hidden_dim,self.drop_prob)
        self.fc_interior_5 = LBR(self.hidden_dim, self.hidden_dim,self.drop_prob)
        self.fc_interior_6 = LBR(self.hidden_dim, self.hidden_dim,self.drop_prob)
        self.fc_interior_7 = nn.Dense(int(np.prod(self.data_shape)))
        self.barycentre=Barycentre(batch_size=self.batch_size,barycenter=self.barycenter)
        

    def __call__(self, z):
        tmp=self.fc_interior_7(self.fc_interior_6(self.fc_interior_5(self.fc_interior_4(self.fc_interior_3(self.fc_interior_2(self.fc_interior_1(z)))))))
        x=self.pca.inverse_transform(tmp)
        result=self.barycentre.apply(x)
        return result
 
