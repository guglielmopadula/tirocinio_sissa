from torch import nn
from models.basic_layers.lbr import LBR 


class Latent_Discriminator_base(nn.Module):
    def __init__(self, latent_dim_1, latent_dim_2, hidden_dim,data_shape,drop_prob):
        super().__init__()
        self.data_shape=data_shape
        self.fc1_interior = LBR(latent_dim_1,hidden_dim,drop_prob)
        self.fc2_interior = LBR(hidden_dim,hidden_dim,drop_prob)
        self.fc3_interior = LBR(hidden_dim,hidden_dim,drop_prob)
        self.fc4_interior = LBR(hidden_dim,hidden_dim,drop_prob)
        self.fc5_interior = LBR(hidden_dim,hidden_dim,drop_prob)
        self.fc6_interior = LBR(hidden_dim,hidden_dim,drop_prob)
        self.fc7_interior = nn.Linear(hidden_dim,1)
        self.fc1_boundary = LBR(latent_dim_2,hidden_dim,drop_prob)
        self.fc2_boundary = LBR(hidden_dim,hidden_dim,drop_prob)
        self.fc3_boundary = LBR(hidden_dim,hidden_dim,drop_prob)
        self.fc4_boundary = LBR(hidden_dim,hidden_dim,drop_prob)
        self.fc5_boundary = LBR(hidden_dim,hidden_dim,drop_prob)
        self.fc6_boundary = LBR(hidden_dim,hidden_dim,drop_prob)
        self.fc7_boundary = nn.Linear(hidden_dim,1)
        self.sigmoid=nn.Sigmoid()
        
    def forward(self,x,y):
        x_hat=self.sigmoid(self.fc7_interior(self.fc6_interior(self.fc5_interior(self.fc4_interior(self.fc3_interior(self.fc2_interior(self.fc1_interior(x))))))))
        y_hat=self.sigmoid(self.fc7_boundary(self.fc6_boundary(self.fc5_boundary(self.fc4_boundary(self.fc3_boundary(self.fc2_boundary(self.fc1_boundary(y))))))))
        return x_hat,y_hat