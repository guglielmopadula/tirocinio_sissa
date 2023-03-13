from models.basic_layers.lt import LT
from models.basic_layers.barycentre import Barycentre

import numpy as np
from torch import nn

class Bayesian_decoder_base(nn.Module):
    def __init__(self, latent_dim, hidden_dim, data_shape,pca,batch_size,barycenter):
        super().__init__()
        self.data_shape=data_shape
        self.pca=pca
        self.batch_size=batch_size
        self.barycenter=barycenter
        self.fc_interior_1 = LT(latent_dim, hidden_dim)
        self.fc_interior_2 = LT(hidden_dim, hidden_dim)
        self.fc_interior_3 = LT(hidden_dim, hidden_dim)
        self.fc_interior_4 = LT(hidden_dim, hidden_dim)
        self.fc_interior_5 = LT(hidden_dim, hidden_dim)
        self.fc_interior_6 = LT(hidden_dim, hidden_dim)
        self.fc_interior_7 = nn.Linear(hidden_dim, int(np.prod(self.data_shape)))
        self.barycentre=Barycentre(batch_size=self.batch_size,barycenter=self.barycenter)
        

    def forward(self, z):
        tmp=self.fc_interior_7(self.fc_interior_6(self.fc_interior_5(self.fc_interior_4(self.fc_interior_3(self.fc_interior_2(self.fc_interior_1(z)))))))
        x=self.pca.inverse_transform(tmp)
        result=self.barycentre(x)
        return result
 
