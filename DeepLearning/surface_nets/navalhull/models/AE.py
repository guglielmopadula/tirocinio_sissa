from pytorch_lightning import LightningModule
from torch import nn
from models.basic_layers.encoder import Encoder_base
from models.basic_layers.decoder import Decoder_base
from models.losses.losses import L2_loss
import torch


class AE(LightningModule):
    
    class Encoder(nn.Module):
        def __init__(self, latent_dim_1,latent_dim_2, hidden_dim,data_shape,pca_1,pca_2,drop_prob):
            super().__init__()
            self.encoder_base=Encoder_base(latent_dim_1=latent_dim_1,latent_dim_2=latent_dim_2, hidden_dim=hidden_dim, data_shape=data_shape,pca_1=pca_1,pca_2=pca_2,drop_prob=drop_prob)
            
        def forward(self,x,y):
            x_hat,y_hat=self.encoder_base(x,y)
            return x_hat,y_hat
        
    class Decoder(nn.Module):

        def __init__(self, latent_dim_1,latent_dim_2, hidden_dim, data_shape,temp_zero,newtriangles_zero,pca_1,pca_2,edge_matrix,vertices_face,cvxpylayer,k,local_indices_1,local_indices_2,drop_prob):
            super().__init__()
            self.decoder_base=Decoder_base(latent_dim_1=latent_dim_1,latent_dim_2=latent_dim_2, hidden_dim=hidden_dim, data_shape=data_shape,local_indices_1=local_indices_1,local_indices_2=local_indices_2,temp_zero=temp_zero,newtriangles_zero=newtriangles_zero,pca_1=pca_1,pca_2=pca_2,edge_matrix=edge_matrix,vertices_face=vertices_face,cvxpylayer=cvxpylayer,k=k,drop_prob=drop_prob)

        def forward(self,x,y):
            return self.decoder_base(x,y)
    
    def __init__(self,data_shape,temp_zero,local_indices_1,local_indices_2,newtriangles_zero,pca_1,pca_2,edge_matrix,vertices_face,cvxpylayer,k,latent_dim_1,latent_dim_2,batch_size,drop_prob,hidden_dim: int= 300,**kwargs):
        super().__init__()
        #self.save_hyperparameters()
        self.temp_zero=temp_zero
        self.newtriangles_zero=newtriangles_zero
        self.pca_1=pca_1
        self.drop_prob=drop_prob
        self.local_indices_1=local_indices_1
        self.local_indices_2=local_indices_2
        self.pca_2=pca_2
        self.edge_matrix=edge_matrix
        self.k=k
        self.latent_dim_1=latent_dim_1
        self.latent_dim_2=latent_dim_2
        self.hidden_dim=hidden_dim
        self.vertices_face=vertices_face
        self.cvxpylayer=cvxpylayer
        # networks
        self.data_shape = data_shape
        self.decoder = self.Decoder(latent_dim_1=self.latent_dim_1,latent_dim_2=self.latent_dim_2,hidden_dim=self.hidden_dim ,data_shape=self.data_shape,local_indices_1=self.local_indices_1,local_indices_2=self.local_indices_2,temp_zero=self.temp_zero,newtriangles_zero=self.newtriangles_zero,pca_1=self.pca_1,pca_2=self.pca_2,edge_matrix=self.edge_matrix,vertices_face=self.vertices_face,cvxpylayer=self.cvxpylayer,k=self.k,drop_prob=self.drop_prob)
        self.encoder = self.Encoder(data_shape=self.data_shape, latent_dim_1=self.latent_dim_1,latent_dim_2=self.latent_dim_2,hidden_dim=self.hidden_dim,pca_1=self.pca_1,pca_2=self.pca_2,drop_prob=self.drop_prob)


    def training_step(self, batch, batch_idx):
        x,y=batch
        z,w=self.encoder(x,y)
        x_hat,y_hat=self.decoder(z,w)
        x_hat=x_hat.reshape(x.shape)
        y_hat=y_hat.reshape(y.shape)
        loss = 0.5*L2_loss(x_hat,x)+0.5*L2_loss(y_hat,y)
        self.log("train_ae_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x,y=batch
        z,w=self.encoder(x,y)
        x_hat,y_hat=self.decoder(z,w)
        x_hat=x_hat.reshape(x.shape)
        y_hat=y_hat.reshape(y.shape)
        loss = 0.5*L2_loss(x_hat,x)+0.5*L2_loss(y_hat,y)
        self.log("validation_ae_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x,y=batch
        z,w=self.encoder(x,y)
        x_hat,y_hat=self.decoder(z,w)
        x_hat=x_hat.reshape(x.shape)
        y_hat=y_hat.reshape(y.shape)
        loss = 0.5*L2_loss(x_hat,x)+0.5*L2_loss(y_hat,y)
        self.log("test_ae_loss", loss)
        return loss

    def get_latent(self,data):
        return self.encoder.forward(data)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return {"optimizer": optimizer}

    def sample_mesh(self,mean=None,var=None):
        if mean==None:
            mean_1=torch.zeros(1,self.latent_dim_1)
            mean_2=torch.zeros(1,self.latent_dim_2)

        if var==None:
            var_1=torch.ones(1,self.latent_dim_1)
            var_2=torch.ones(1,self.latent_dim_2)

        z = torch.sqrt(var_1)*torch.randn(1,self.latent_dim_1)+mean_1
        w = torch.sqrt(var_2)*torch.randn(1,self.latent_dim_2)+mean_2
        temp_interior,temp_boundary=self.decoder(z,w)
        return temp_interior,temp_boundary
     
