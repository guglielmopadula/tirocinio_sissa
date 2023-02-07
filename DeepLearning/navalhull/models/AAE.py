from pytorch_lightning import LightningModule
from torch import nn
from models.basic_layers.encoder import Encoder_base
from models.basic_layers.decoder import Decoder_base
from models.basic_layers.ld import Latent_Discriminator_base
import itertools
from models.losses.losses import L2_loss,CE_loss
import torch

class AAE(LightningModule):
    
    

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
    
    
    class Discriminator(nn.Module):
        def __init__(self, latent_dim,hidden_dim,drop_prob):
            super().__init__()
            self.discriminator=Latent_Discriminator_base(latent_dim=latent_dim,hidden_dim=hidden_dim, drop_prob=drop_prob)
             
        def forward(self,x):
            x_hat=self.discriminator(x)
            return x_hat



    def __init__(self,data_shape,reduced_data_shape,temp_zero,local_indices_1,local_indices_2,newtriangles_zero,pca,edge_matrix,vertices_face_x,vertices_face_xy,k,latent_dim,batch_size,drop_prob,ae_hyp=0.9999,hidden_dim: int= 300,**kwargs):
        super().__init__()
        self.temp_zero=temp_zero
        self.newtriangles_zero=newtriangles_zero
        self.pca=pca
        self.local_indices_1=local_indices_1
        self.local_indices_2=local_indices_2
        self.edge_matrix=edge_matrix
        self.k=k
        self.reduced_data_shape=reduced_data_shape
        self.ae_hyp=ae_hyp
        self.latent_dim=latent_dim
        self.hidden_dim=hidden_dim
        self.vertices_face_x=vertices_face_x
        self.vertices_face_xy=vertices_face_xy
        self.data_shape = data_shape
        self.decoder = self.Decoder(latent_dim=self.latent_dim,hidden_dim=self.hidden_dim,reduced_data_shape=self.reduced_data_shape ,data_shape=self.data_shape,local_indices_1=self.local_indices_1,local_indices_2=self.local_indices_2,temp_zero=self.temp_zero,newtriangles_zero=self.newtriangles_zero,pca=self.pca,edge_matrix=self.edge_matrix,vertices_face_x=self.vertices_face_x,vertices_face_xy=self.vertices_face_xy,k=self.k,drop_prob=drop_prob)
        self.encoder = self.Encoder( latent_dim=self.latent_dim,reduced_data_shape=self.reduced_data_shape,hidden_dim=self.hidden_dim,pca=self.pca,drop_prob=drop_prob)
        self.discriminator=self.Discriminator(latent_dim=self.latent_dim,hidden_dim=self.hidden_dim,drop_prob=drop_prob)
    

    def training_step(self, batch, batch_idx, optimizer_idx ):
        x=batch
        z_enc=self.encoder(x)
        z=torch.randn(len(x), self.latent_dim).type_as(x)
        x_disc_e=self.discriminator(z_enc)
        x_disc=self.discriminator(z)
        x_hat=self.decoder(z_enc)
        x_hat=x_hat.reshape(x.shape)


        if optimizer_idx==0:
            ae_loss = self.ae_hyp*L2_loss(x_hat,x)+(1-self.ae_hyp)*CE_loss(x_disc_e,torch.ones_like(x_disc_e)).mean()
            self.log("train_ae_loss", ae_loss)
            return ae_loss
        
        if optimizer_idx==1:
            real_loss = CE_loss(x_disc,torch.ones_like(x_disc)).mean()
            fake_loss = CE_loss(x_disc_e,torch.zeros_like(x_disc_e)).mean()
            tot_loss= (real_loss+fake_loss)/2
            self.log("train_aee_loss", tot_loss)
            return tot_loss
                    
    def get_latent(self,data):
        return self.encoder.forward(data)
    
    def test_step(self, batch, batch_idx):
        x=batch
        z_enc=self.encoder(x)
        x_hat=self.decoder(z_enc)
        x_hat=x_hat.reshape(x.shape)
        ae_loss = self.ae_hyp*L2_loss(x_hat,x)
        self.log("test_aae_loss", ae_loss)
        return ae_loss


    def configure_optimizers(self):
        optimizer_ae = torch.optim.AdamW(itertools.chain(self.encoder.parameters(), self.decoder.parameters()), lr=4e-3)
        optimizer_disc = torch.optim.AdamW(self.discriminator.parameters(), lr=1e-3)
        return [optimizer_ae,optimizer_disc], []
    def sample_mesh(self,mean=None,var=None):
        device=self.decoder.decoder_base.pca._V.device
        self=self.to(device)
        if mean==None:
            mean_1=torch.zeros(1,self.latent_dim)

        if var==None:
            var_1=torch.ones(1,self.latent_dim)

        z = torch.sqrt(var_1)*torch.randn(1,self.latent_dim)+mean_1
        z=z.to(device)
        tmp=self.decoder(z)
        return tmp
     
