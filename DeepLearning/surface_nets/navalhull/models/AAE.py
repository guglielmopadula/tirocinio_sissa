from pytorch_lightning import LightningModule
from torch import nn
from models.basic_layers.encoder import Encoder_base
from models.basic_layers.decoder import Decoder_base
from models.basic_layers.ld import Latent_Discriminator_base
import itertools
from models.losses.losses import L2_loss
import torch

class AAE(LightningModule):
    
    

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
    
    class Discriminator(nn.Module):
        def __init__(self, latent_dim_1,latent_dim_2, hidden_dim,data_shape,drop_prob):
            super().__init__()
            self.discriminator=Latent_Discriminator_base(latent_dim_1=latent_dim_1, data_shape=data_shape,hidden_dim=hidden_dim, latent_dim_2=latent_dim_2,drop_prob=drop_prob)
             
        def forward(self,x,y):
            x_hat,y_hat=self.discriminator(x,y)
            return x_hat,y_hat



    def __init__(self,data_shape,temp_zero,local_indices_1,local_indices_2,newtriangles_zero,pca_1,pca_2,edge_matrix,vertices_face,cvxpylayer,k,latent_dim_1,latent_dim_2,batch_size,drop_prob,ae_hyp=0.999,hidden_dim: int= 300,**kwargs):
        super().__init__()
        self.temp_zero=temp_zero
        self.newtriangles_zero=newtriangles_zero
        self.pca_1=pca_1
        self.local_indices_1=local_indices_1
        self.local_indices_2=local_indices_2
        self.pca_2=pca_2
        self.edge_matrix=edge_matrix
        self.k=k
        self.ae_hyp=ae_hyp
        self.latent_dim_1=latent_dim_1
        self.latent_dim_2=latent_dim_2
        self.hidden_dim=hidden_dim
        self.vertices_face=vertices_face
        self.cvxpylayer=cvxpylayer
        # networks
        self.data_shape = data_shape
        self.decoder = self.Decoder(latent_dim_1=self.latent_dim_1,latent_dim_2=self.latent_dim_2,hidden_dim=self.hidden_dim ,data_shape=self.data_shape,local_indices_1=self.local_indices_1,local_indices_2=self.local_indices_2,temp_zero=self.temp_zero,newtriangles_zero=self.newtriangles_zero,pca_1=self.pca_1,pca_2=self.pca_2,edge_matrix=self.edge_matrix,vertices_face=self.vertices_face,cvxpylayer=self.cvxpylayer,k=self.k,drop_prob=drop_prob)
        self.encoder = self.Encoder(data_shape=self.data_shape, latent_dim_1=self.latent_dim_1,latent_dim_2=self.latent_dim_2,hidden_dim=self.hidden_dim,pca_1=self.pca_1,pca_2=self.pca_2,drop_prob=drop_prob)
        self.discriminator=self.Discriminator(data_shape=self.data_shape, latent_dim_1=self.latent_dim_1,latent_dim_2=self.latent_dim_2 ,hidden_dim=self.hidden_dim,drop_prob=drop_prob)
    

    def training_step(self, batch, batch_idx, optimizer_idx ):
        x,y=batch
        z_enc_1,z_enc_2=self.encoder(x,y)
        z_1=torch.randn(len(x), self.latent_dim_1).type_as(x)
        z_2=torch.randn(len(y), self.latent_dim_2).type_as(y)
        x_disc_e,y_disc_e=self.discriminator(z_enc_1,z_enc_2)
        x_disc,y_disc=self.discriminator(z_1,z_2)
        x_hat,y_hat=self.decoder(z_enc_1,z_enc_2)
        x_hat=x_hat.reshape(x.shape)
        y_hat=y_hat.reshape(y.shape)


        if optimizer_idx==0:
            ae_loss = 0.5*self.ae_hyp*(L2_loss(x_hat,x)+L2_loss(y_hat,y))-0.5*(1-self.ae_hyp)*(x_disc_e+y_disc_e).mean()
            self.log("train_ae_loss", ae_loss)
            return ae_loss
        
        if optimizer_idx==1:
            real_loss = 0.5*(x_disc_e+y_disc_e).mean()
            fake_loss = -0.5*(x_disc+y_disc).mean()   
            tot_loss= (real_loss+fake_loss)/2
            self.log("train_aee_loss", tot_loss)
            return tot_loss
            
    def validation_step(self, batch, batch_idx):
        x,y=batch
        z_enc_1,z_enc_2=self.encoder(x,y)
        z_1=torch.randn(len(x), self.latent_dim_1).type_as(x)
        z_2=torch.randn(len(y), self.latent_dim_2).type_as(y)
        x_disc_e,y_disc_e=self.discriminator(z_enc_1,z_enc_2)
        x_disc,y_disc=self.discriminator(z_1,z_2)
        x_hat,y_hat=self.decoder(z_enc_1,z_enc_2)
        x_hat=x_hat.reshape(x.shape)
        y_hat=y_hat.reshape(y.shape)
        ae_loss = self.ae_hyp*(L2_loss(x_hat,x)+L2_loss(y_hat,y))
        self.log("val_aae_loss", ae_loss)
        return ae_loss
        
        
    def get_latent(self,data):
        return self.encoder.forward(data)
    
    def test_step(self, batch, batch_idx):
        x,y=batch
        z_enc_1,z_enc_2=self.encoder(x,y)
        z_1=torch.randn(len(x), self.latent_dim_1).type_as(x)
        z_2=torch.randn(len(y), self.latent_dim_2).type_as(y)
        x_disc_e,y_disc_e=self.discriminator(z_enc_1,z_enc_2)
        x_disc,y_disc=self.discriminator(z_1,z_2)
        x_hat,y_hat=self.decoder(z_enc_1,z_enc_2)
        x_hat=x_hat.reshape(x.shape)
        y_hat=y_hat.reshape(y.shape)
        ae_loss = self.ae_hyp*(L2_loss(x_hat,x)+L2_loss(y_hat,y))
        self.log("test_aae_loss", ae_loss)
        return ae_loss


    def configure_optimizers(self):
        optimizer_ae = torch.optim.AdamW(itertools.chain(self.encoder.parameters(), self.decoder.parameters()), lr=1e-3)
        optimizer_disc = torch.optim.AdamW(self.discriminator.parameters(), lr=1e-3)
        return [optimizer_ae,optimizer_disc], []
    
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

