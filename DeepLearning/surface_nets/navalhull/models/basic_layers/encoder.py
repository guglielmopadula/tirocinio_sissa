from models.basic_layers.lbr import LBR
import numpy as np
from torch import nn
class Encoder_base(nn.Module):
    def __init__(self, latent_dim_1,latent_dim_2, hidden_dim,data_shape,pca_1,pca_2,drop_prob):
        super().__init__()
        self.data_shape=data_shape
        self.latent_dim_1=latent_dim_1
        self.latent_dim_2=latent_dim_2
        self.pca_1=pca_1
        self.pca_2=pca_2
        self.drop_prob=drop_prob
        self.fc_interior_1 = LBR(int(np.prod(self.data_shape[0])), hidden_dim,drop_prob)
        self.fc_interior_2 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_interior_3 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_interior_4 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_interior_5 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_interior_6 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_interior_7 = nn.Linear(hidden_dim, latent_dim_1)
        self.fc_boundary_1 = LBR(int(np.prod(self.data_shape[1])), hidden_dim,drop_prob)
        self.fc_boundary_2 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_boundary_3 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_boundary_4 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_boundary_5 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_boundary_6 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_boundary_7 = nn.Linear(hidden_dim, latent_dim_2)

        self.tanh=nn.Tanh()
        self.batch_mu_1=nn.BatchNorm1d(self.latent_dim_1,affine=False,track_running_stats=False)
        self.batch_mu_2=nn.BatchNorm1d(self.latent_dim_2,affine=False,track_running_stats=False)


    def forward(self, x,y):
        x=x.reshape(x.size(0),-1)
        x=self.pca_1.transform(x)
        mu_1=self.fc_interior_7(self.fc_interior_6(self.fc_interior_5(self.fc_interior_4(self.fc_interior_3(self.fc_interior_2(self.fc_interior_1(x)))))))
        mu_1=self.batch_mu_1(mu_1)
        y=y.reshape(y.size(0),-1)
        y=self.pca_2.transform(y)
        mu_2=self.fc_boundary_7(self.fc_boundary_6(self.fc_boundary_5(self.fc_boundary_4(self.fc_boundary_3(self.fc_boundary_2(self.fc_boundary_1(y)))))))
        mu_2=self.batch_mu_2(mu_2)
        return mu_1,mu_2
 
