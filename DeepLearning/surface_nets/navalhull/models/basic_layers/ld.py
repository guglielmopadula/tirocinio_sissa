from torch import nn
from models.basic_layers.lbr import LBR 


class Latent_Discriminator_base(nn.Module):
    def __init__(self, latent_dim, hidden_dim,drop_prob):
        super().__init__()
        self.fc1_interior = LBR(latent_dim,hidden_dim,drop_prob)
        self.fc2_interior = LBR(hidden_dim,hidden_dim,drop_prob)
        self.fc3_interior = LBR(hidden_dim,hidden_dim,drop_prob)
        self.fc4_interior = LBR(hidden_dim,hidden_dim,drop_prob)
        self.fc5_interior = LBR(hidden_dim,hidden_dim,drop_prob)
        self.fc6_interior = LBR(hidden_dim,hidden_dim,drop_prob)
        self.fc7_interior = nn.Linear(hidden_dim,1)
        self.sigmoid=nn.Sigmoid()
        
    def forward(self,x):
        x_hat=self.sigmoid(self.fc7_interior(self.fc6_interior(self.fc5_interior(self.fc4_interior(self.fc3_interior(self.fc2_interior(self.fc1_interior(x))))))))
        return x_hat