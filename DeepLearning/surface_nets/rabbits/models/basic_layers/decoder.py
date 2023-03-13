from models.basic_layers.lbr import LBR
from models.basic_layers.barycentre import Barycentre

import numpy as np
from torch import nn

class Decoder_base(nn.Module):
    def __init__(self, latent_dim, hidden_dim, data_shape,pca,batch_size,drop_prob,barycenter):
        super().__init__()
        self.data_shape=data_shape
        self.pca=pca
        self.batch_size=batch_size
        self.barycenter=barycenter
        self.drop_prob=drop_prob
        self.fc_interior_1 = LBR(latent_dim, hidden_dim,drop_prob)
        self.fc_interior_2 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_interior_3 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_interior_4 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_interior_5 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_interior_6 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_interior_7 = nn.Linear(hidden_dim, int(np.prod(self.data_shape)))
        self.barycentre=Barycentre(batch_size=self.batch_size,barycenter=self.barycenter)
        

    def forward(self, z):
        tmp=self.fc_interior_7(self.fc_interior_6(self.fc_interior_5(self.fc_interior_4(self.fc_interior_3(self.fc_interior_2(self.fc_interior_1(z)))))))
        x=self.pca.inverse_transform(tmp)
        result=self.barycentre(x)
        return result
 
