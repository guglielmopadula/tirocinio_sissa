from pytorch_lightning import LightningModule
from torch import nn
from models.basic_layers.encoder import Encoder_base
from models.basic_layers.decoder import Decoder_base
from models.basic_layers.variance_estimator import Variance_estimator
from models.losses.losses import L2_loss
import torch

class VAE(LightningModule):
    
    class Encoder(nn.Module):
        def __init__(self, latent_dim, hidden_dim,reduced_data_shape,pca,drop_prob):
            super().__init__()
            self.encoder_base=Encoder_base(latent_dim=latent_dim,hidden_dim=hidden_dim, reduced_data_shape=reduced_data_shape,pca=pca,drop_prob=drop_prob)
            self.variance_estimator=Variance_estimator(latent_dim, hidden_dim,drop_prob=drop_prob)
            
        def forward(self,x):
            mu=self.encoder_base(x)
            sigma=self.variance_estimator(mu)
            return mu,sigma
        
    class Decoder(nn.Module):

        def __init__(self, latent_dim, hidden_dim, reduced_data_shape,data_shape,temp_zero,newtriangles_zero,pca,edge_matrix,vertices_face_x,vertices_face_xy,k,local_indices_1,local_indices_2,drop_prob):
            super().__init__()
            self.decoder_base=Decoder_base(latent_dim=latent_dim, hidden_dim=hidden_dim, data_shape=data_shape,local_indices_1=local_indices_1,local_indices_2=local_indices_2,temp_zero=temp_zero,newtriangles_zero=newtriangles_zero,pca=pca,edge_matrix=edge_matrix,vertices_face_x=vertices_face_x,vertices_face_xy=vertices_face_xy,k=k,drop_prob=drop_prob,reduced_data_shape=reduced_data_shape)

        def forward(self,x):
            return self.decoder_base(x)


    def __init__(self,data_shape,temp_zero,local_indices_1,local_indices_2,newtriangles_zero,pca,edge_matrix,vertices_face_x,vertices_face_xy,k,latent_dim,batch_size,drop_prob,reduced_data_shape,beta=0.01,hidden_dim: int= 300,**kwargs):
        super().__init__()
        self.temp_zero=temp_zero
        self.newtriangles_zero=newtriangles_zero
        self.pca=pca
        self.local_indices_1=local_indices_1
        self.local_indices_2=local_indices_2
        self.log_scale=nn.Parameter(torch.Tensor([0.0]))
        self.edge_matrix=edge_matrix
        self.k=k
        self.beta=beta
        self.batch_size=batch_size
        self.drop_prob=drop_prob
        self.reduced_data_shape=reduced_data_shape
        self.latent_dim=latent_dim
        self.hidden_dim=hidden_dim
        self.vertices_face_x=vertices_face_x
        self.vertices_face_xy=vertices_face_xy
        self.data_shape = data_shape
        self.decoder = self.Decoder(latent_dim=self.latent_dim,hidden_dim=self.hidden_dim ,data_shape=self.data_shape,local_indices_1=self.local_indices_1,local_indices_2=self.local_indices_2,temp_zero=self.temp_zero,newtriangles_zero=self.newtriangles_zero,pca=self.pca,edge_matrix=self.edge_matrix,vertices_face_x=self.vertices_face_x,vertices_face_xy=self.vertices_face_xy,k=self.k,drop_prob=self.drop_prob, reduced_data_shape=self.reduced_data_shape)
        self.encoder = self.Encoder(reduced_data_shape=self.reduced_data_shape, latent_dim=self.latent_dim,hidden_dim=self.hidden_dim,pca=self.pca,drop_prob=self.drop_prob)
        
    
    def training_step(self, batch, batch_idx):
        x=batch
        mu,sigma= self.encoder(x)
        q_1 = torch.distributions.Normal(mu.reshape(self.batch_size,-1), sigma.reshape(self.batch_size,-1))
        standard_1=torch.distributions.Normal(torch.zeros_like(mu.reshape(self.batch_size,-1)), torch.ones_like(sigma.reshape(self.batch_size,-1)))
        z1_sampled = q_1.rsample()
        x_hat = self.decoder(z1_sampled)
        p_1=torch.distributions.Normal(x_hat.reshape(self.batch_size,-1),torch.exp(self.log_scale))
        reconstruction=p_1.log_prob(x.reshape(self.batch_size,-1)).sum(dim=1)
        reg=torch.distributions.kl_divergence(q_1, standard_1).sum(dim=1)
        elbo=(reconstruction-self.beta*reg).mean(dim=0)
        self.log("train_vae_loss", -elbo)
        return -elbo
    
    def validation_step(self, batch, batch_idx):
        x=batch
        z=self.sample_mesh().reshape(1,-1)
        loss=torch.min(torch.linalg.norm((x-z),axis=1))/torch.linalg.norm(x)
        self.log("val_rec", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x=batch
        mu,sigma= self.encoder(x)
        q_1 = torch.distributions.Normal(mu.reshape(self.batch_size,-1), sigma.reshape(self.batch_size,-1))
        standard_1=torch.distributions.Normal(torch.zeros_like(mu.reshape(self.batch_size,-1)), torch.ones_like(sigma.reshape(self.batch_size,-1)))
        z1_sampled = q_1.rsample()
        x_hat = self.decoder(z1_sampled)
        p_1=torch.distributions.Normal(x_hat.reshape(self.batch_size,-1),torch.exp(self.log_scale))
        reconstruction=p_1.log_prob(x.reshape(self.batch_size,-1)).mean(dim=1)
        reg=torch.distributions.kl_divergence(q_1, standard_1).mean(dim=1)
        elbo=(reconstruction-self.beta*reg).mean(dim=0)
        self.log("train_vae_loss", -elbo)
        return -elbo
    
    def get_latent(self,data):
        return self.encoder.forward(data)[0]


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        return {"optimizer": optimizer}
    
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
     
