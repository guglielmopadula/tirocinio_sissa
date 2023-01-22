from pytorch_lightning import LightningModule
from torch import nn
from models.basic_layers.encoder import Encoder_base
from models.basic_layers.decoder import Decoder_base
from models.basic_layers.variance_estimator import Variance_estimator
from models.losses.losses import L2_loss
import torch

class VAE(LightningModule):
    
    class Encoder(nn.Module):
        def __init__(self, latent_dim_1,latent_dim_2, hidden_dim,data_shape,pca_1,pca_2,drop_prob):
            super().__init__()
            self.encoder_base=Encoder_base(latent_dim_1=latent_dim_1,latent_dim_2=latent_dim_2, hidden_dim=hidden_dim, data_shape=data_shape,pca_1=pca_1,pca_2=pca_2,drop_prob=drop_prob)
            self.variance_estimator=Variance_estimator(latent_dim_1,latent_dim_2, hidden_dim, data_shape,drop_prob=drop_prob)
            
        def forward(self,x,y):
            mu_1,mu_2=self.encoder_base(x,y)
            sigma_1,sigma_2=self.variance_estimator(mu_1,mu_2)
            return mu_1,mu_2,sigma_1,sigma_2
        
    class Decoder(nn.Module):

        def __init__(self, latent_dim_1,latent_dim_2, hidden_dim, data_shape,temp_zero,newtriangles_zero,pca_1,pca_2,edge_matrix,vertices_face,cvxpylayer,k,local_indices_1,local_indices_2,drop_prob):
            super().__init__()
            self.decoder_base=Decoder_base(latent_dim_1=latent_dim_1,latent_dim_2=latent_dim_2, hidden_dim=hidden_dim, data_shape=data_shape,local_indices_1=local_indices_1,local_indices_2=local_indices_2,temp_zero=temp_zero,newtriangles_zero=newtriangles_zero,pca_1=pca_1,pca_2=pca_2,edge_matrix=edge_matrix,vertices_face=vertices_face,cvxpylayer=cvxpylayer,k=k,drop_prob=drop_prob)

        def forward(self,x,y):
            return self.decoder_base(x,y)


    def __init__(self,data_shape,temp_zero,local_indices_1,local_indices_2,newtriangles_zero,pca_1,pca_2,edge_matrix,vertices_face,cvxpylayer,k,latent_dim_1,latent_dim_2,batch_size,drop_prob,beta=0.01,hidden_dim: int= 300,**kwargs):
        super().__init__()
        #self.save_hyperparameters()
        self.temp_zero=temp_zero
        self.newtriangles_zero=newtriangles_zero
        self.pca_1=pca_1
        self.local_indices_1=local_indices_1
        self.local_indices_2=local_indices_2
        self.pca_2=pca_2
        self.log_scale=nn.Parameter(torch.Tensor([0.0]))
        self.edge_matrix=edge_matrix
        self.k=k
        self.beta=beta
        self.batch_size=batch_size
        self.drop_prob=drop_prob
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
        mu_1,mu_2,sigma_1,sigma_2 = self.encoder(x,y)
        q_1 = torch.distributions.Normal(mu_1.reshape(self.batch_size,-1), sigma_1.reshape(self.batch_size,-1))
        q_2 = torch.distributions.Normal(mu_2.reshape(self.batch_size,-1), sigma_2.reshape(self.batch_size,-1))
        standard_1=torch.distributions.Normal(torch.zeros_like(mu_1.reshape(self.batch_size,-1)), torch.ones_like(sigma_1.reshape(self.batch_size,-1)))
        standard_2=torch.distributions.Normal(torch.zeros_like(mu_2.reshape(self.batch_size,-1)), torch.ones_like(sigma_2.reshape(self.batch_size,-1)))
        z1_sampled = q_1.rsample()
        z2_sampled = q_2.rsample()
        x_hat,y_hat = self.decoder(z1_sampled,z2_sampled)
        p_1=torch.distributions.Normal(x_hat.reshape(self.batch_size,-1),torch.exp(self.log_scale))
        p_2=torch.distributions.Normal(y_hat.reshape(self.batch_size,-1),torch.exp(self.log_scale))
        reconstruction=0.5*p_1.log_prob(x.reshape(self.batch_size,-1)).mean(dim=1)+0.5*p_2.log_prob(y.reshape(self.batch_size,-1)).mean(dim=1)
        reg=0.5*torch.distributions.kl_divergence(q_1, standard_1).mean(dim=1)+0.5*torch.distributions.kl_divergence(q_2, standard_2).mean(dim=1)
        elbo=(reconstruction-self.beta*reg).mean(dim=0)
        self.log("train_vae_loss", -elbo)
        return -elbo
    
    
    def get_latent(self,data):
        return self.encoder.forward(data)[0]

    
    def validation_step(self, batch, batch_idx):
        x,y=batch
        mu_1,mu_2,sigma_1,sigma_2 = self.encoder(x,y)
        q_1 = torch.distributions.Normal(mu_1, sigma_1)
        q_2 = torch.distributions.Normal(mu_2, sigma_2)
        standard_1=torch.distributions.Normal(torch.zeros_like(mu_1), torch.ones_like(sigma_1))
        standard_2=torch.distributions.Normal(torch.zeros_like(mu_2), torch.ones_like(sigma_2))
        z1_sampled = q_1.rsample()
        z2_sampled = q_2.rsample()
        x_hat,y_hat = self.decoder(z1_sampled,z2_sampled)
        x_hat=x_hat.reshape(x.shape)
        y_hat=y_hat.reshape(y.shape)

        loss=0.5*L2_loss(x_hat, x)+0.5*L2_loss(y_hat, y)
        reg=0.5*torch.distributions.kl_divergence(q_1, standard_1).mean()+0.5*torch.distributions.kl_divergence(q_2, standard_2).mean()
        self.log("val_vae_loss", loss)
        return loss+reg

    def test_step(self, batch, batch_idx):
        x,y=batch
        mu_1,mu_2,sigma_1,sigma_2 = self.encoder(x,y)
        q_1 = torch.distributions.Normal(mu_1, sigma_1)
        q_2 = torch.distributions.Normal(mu_2, sigma_2)
        standard_1=torch.distributions.Normal(torch.zeros_like(mu_1), torch.ones_like(sigma_1))
        standard_2=torch.distributions.Normal(torch.zeros_like(mu_2), torch.ones_like(sigma_2))
        z1_sampled = q_1.rsample()
        z2_sampled = q_2.rsample()
        x_hat,y_hat = self.decoder(z1_sampled,z2_sampled)
        x_hat=x_hat.reshape(x.shape)
        y_hat=y_hat.reshape(y.shape)

        loss=0.5*L2_loss(x_hat, x)+0.5*L2_loss(y_hat, y)
        reg=0.5*torch.distributions.kl_divergence(q_1, standard_1).mean()+0.5*torch.distributions.kl_divergence(q_2, standard_2).mean()
        self.log("test_vae_loss", loss)
        return loss+reg

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return {"optimizer": optimizer}
    def sample_mesh(self,mean=None,var=None):
        device=self.decoder.decoder_base.pca_1._V.device
        self=self.to(device)
        if mean==None:
            mean_1=torch.zeros(1,self.latent_dim_1)
            mean_2=torch.zeros(1,self.latent_dim_2)

        if var==None:
            var_1=torch.ones(1,self.latent_dim_1)
            var_2=torch.ones(1,self.latent_dim_2)

        z = torch.sqrt(var_1)*torch.randn(1,self.latent_dim_1)+mean_1
        w = torch.sqrt(var_2)*torch.randn(1,self.latent_dim_2)+mean_2
        w=w.to(device)
        z=z.to(device)
        temp_interior,temp_boundary=self.decoder(z,w)
        return temp_interior,temp_boundary

