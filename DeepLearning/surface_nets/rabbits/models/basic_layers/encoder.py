from models.basic_layers.lbr import LBR
import numpy as np
from torch import nn
class Encoder_base(nn.Module):
    def __init__(self, latent_dim,hidden_dim,data_shape,pca,drop_prob,batch_size):
        super().__init__()
        self.data_shape=data_shape
        self.batch_size=batch_size
        self.latent_dim=latent_dim
        self.pca=pca
        self.drop_prob=drop_prob
        self.fc_interior_1 = LBR(int(np.prod(self.data_shape)), hidden_dim,drop_prob)
        self.fc_interior_2 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_interior_3 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_interior_4 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_interior_5 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_interior_6 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_interior_7 = nn.Linear(hidden_dim, latent_dim)
        self.tanh=nn.Tanh()
        self.batch_mu_1=nn.BatchNorm1d(self.latent_dim,affine=False,track_running_stats=False)


    def forward(self, x):
        x=x.reshape(self.batch_size,-1)
        x=self.pca.transform(x)
        mu_1=self.fc_interior_7(self.fc_interior_6(self.fc_interior_5(self.fc_interior_4(self.fc_interior_3(self.fc_interior_2(self.fc_interior_1(x)))))))
        mu_1=self.batch_mu_1(mu_1)
        return mu_1
 
