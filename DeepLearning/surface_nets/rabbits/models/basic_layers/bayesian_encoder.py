from models.basic_layers.lt import LT
import numpy as np
from torch import nn
class Bayesian_encoder_base(nn.Module):
    def __init__(self, latent_dim,hidden_dim,data_shape,pca,batch_size):
        super().__init__()
        self.data_shape=data_shape
        self.batch_size=batch_size
        self.latent_dim=latent_dim
        self.pca=pca
        self.fc_interior_1 = LT(int(np.prod(self.data_shape)), hidden_dim)
        self.fc_interior_2 = LT(hidden_dim, hidden_dim)
        self.fc_interior_3 = LT(hidden_dim, hidden_dim)
        self.fc_interior_4 = LT(hidden_dim, hidden_dim)
        self.fc_interior_5 = LT(hidden_dim, hidden_dim)
        self.fc_interior_6 = LT(hidden_dim, hidden_dim)
        self.fc_interior_7 = nn.Linear(hidden_dim, latent_dim)
        self.tanh=nn.Tanh()


    def forward(self, x):
        x=x.reshape(self.batch_size,-1)
        x=self.pca.transform(x)
        mu_1=self.fc_interior_7(self.fc_interior_6(self.fc_interior_5(self.fc_interior_4(self.fc_interior_3(self.fc_interior_2(self.fc_interior_1(x)))))))
        return mu_1
 
