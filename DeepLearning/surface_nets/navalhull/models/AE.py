from pytorch_lightning import LightningModule
from torch import nn
from models.basic_layers.encoder import Encoder_base
from models.basic_layers.decoder import Decoder_base
from models.losses.losses import L2_loss,LI_loss,torch_mmd
import torch


class AE(LightningModule):
    
    class Encoder(nn.Module):
        def __init__(self, latent_dim, hidden_dim,reduced_data_shape,pca,drop_prob):
            super().__init__()
            self.encoder_base=Encoder_base(latent_dim=latent_dim,hidden_dim=hidden_dim, reduced_data_shape=reduced_data_shape,pca=pca,drop_prob=drop_prob)
            
        def forward(self,x):
            x_hat=self.encoder_base(x)
            return x_hat
        
    class Decoder(nn.Module):

        def __init__(self, latent_dim, hidden_dim, data_shape,temp_zero,newtriangles_zero,pca,edge_matrix,vertices_face_x,vertices_face_xy,k,local_indices_1,local_indices_2,drop_prob,reduced_data_shape):
            super().__init__()
            self.decoder_base=Decoder_base(latent_dim=latent_dim, hidden_dim=hidden_dim, data_shape=data_shape,local_indices_1=local_indices_1,local_indices_2=local_indices_2,temp_zero=temp_zero,newtriangles_zero=newtriangles_zero,pca=pca,edge_matrix=edge_matrix,vertices_face_x=vertices_face_x,vertices_face_xy=vertices_face_xy,k=k,drop_prob=drop_prob,reduced_data_shape=reduced_data_shape)

        def forward(self,x):
            return self.decoder_base(x)
    
    def __init__(self,data_shape,reduced_data_shape,temp_zero,local_indices_1,local_indices_2,newtriangles_zero,pca,edge_matrix,vertices_face_x,vertices_face_xy,k,latent_dim,batch_size,drop_prob,hidden_dim: int= 500,**kwargs):
        super().__init__()
        self.temp_zero=temp_zero
        self.newtriangles_zero=newtriangles_zero
        self.pca=pca
        self.drop_prob=drop_prob
        self.local_indices_1=local_indices_1
        self.local_indices_2=local_indices_2
        self.edge_matrix=edge_matrix
        self.reduced_data_shape=reduced_data_shape
        self.k=k
        self.latent_dim=latent_dim
        self.hidden_dim=hidden_dim
        self.vertices_face_x=vertices_face_x
        self.vertices_face_xy=vertices_face_xy
        self.data_shape = data_shape
        self.decoder = self.Decoder(latent_dim=self.latent_dim,hidden_dim=self.hidden_dim ,data_shape=self.data_shape,local_indices_1=self.local_indices_1,local_indices_2=self.local_indices_2,temp_zero=self.temp_zero,newtriangles_zero=self.newtriangles_zero,pca=self.pca,edge_matrix=self.edge_matrix,vertices_face_x=self.vertices_face_x,vertices_face_xy=self.vertices_face_xy,k=self.k,drop_prob=self.drop_prob,reduced_data_shape=self.reduced_data_shape)
        self.encoder = self.Encoder(reduced_data_shape=self.reduced_data_shape, latent_dim=self.latent_dim,hidden_dim=self.hidden_dim,pca=self.pca,drop_prob=self.drop_prob)


    def training_step(self, batch, batch_idx):
        x=batch
        z=self.encoder(x)
        x_hat=self.decoder(z)
        loss = L2_loss(x_hat,x)
        self.log("train_ae_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x=batch
        z=self.sample_mesh(torch.zeros(100,self.latent_dim),torch.ones(100,self.latent_dim))
        loss=L2_loss(batch,z)
        self.log("val_mmd", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x=batch
        z=self.encoder(x)
        x_hat=self.decoder(z)
        loss = L2_loss(x_hat,x)
        self.log("test_ae_loss", loss)
        return loss

    def get_latent(self,data):
        return self.encoder.forward(data)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.00005)
        return {"optimizer": optimizer}

    def sample_mesh(self,mean=None,var=None):
        device=self.decoder.decoder_base.pca._V.device
        self=self.to(device)
        if mean==None:
            mean=torch.zeros(1,self.latent_dim)

        if var==None:
            var=torch.ones(1,self.latent_dim)

        z = torch.sqrt(var)*torch.randn(1,self.latent_dim)+mean
        z=z.to(device)
        tmp=self.decoder(z)
        return tmp
     
