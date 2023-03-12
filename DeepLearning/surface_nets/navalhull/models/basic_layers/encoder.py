from models.basic_layers.lbr import LBR
import numpy as np
from torch import nn
class Encoder_base(nn.Module):
    def __init__(self, latent_dim, hidden_dim,reduced_data_shape,pca,drop_prob):
        super().__init__()
        self.reduced_data_shape=reduced_data_shape
        self.latent_dim=latent_dim
        self.pca=pca
        self.drop_prob=drop_prob
        self.fc_interior_1 = LBR(reduced_data_shape, hidden_dim,drop_prob)
        self.fc_interior_2 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_interior_3 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_interior_4 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_interior_5 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_interior_6 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_interior_6 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_interior_7 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_interior_8 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_interior_9 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_interior_10 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_interior_11 = nn.Linear(hidden_dim, latent_dim)
        self.batch_mu_1=nn.BatchNorm1d(self.latent_dim)


    def forward(self, x):
        x=x.reshape(x.size(0),-1)
        x=self.pca.transform(x)
        mu=self.fc_interior_11(self.fc_interior_10(self.fc_interior_9(self.fc_interior_8(self.fc_interior_7(self.fc_interior_6(self.fc_interior_5(self.fc_interior_4(self.fc_interior_3(self.fc_interior_2(self.fc_interior_1(x)))))))))))
        mu=self.batch_mu_1(mu)
        return mu