from pytorch_lightning import LightningModule
from torch import nn
from models.basic_layers.encoder import Encoder_base
from models.basic_layers.decoder import Decoder_base
from models.basic_layers.variance_estimator import Variance_estimator
from models.losses.losses import L2_loss
import torch

class VAE(LightningModule):
    
    class Encoder(nn.Module):
        def __init__(self, latent_dim, hidden_dim,data_shape,pca,drop_prob,batch_size):
            super().__init__()
            self.encoder_base=Encoder_base(latent_dim=latent_dim, hidden_dim=hidden_dim, data_shape=data_shape,pca=pca,drop_prob=drop_prob,batch_size=batch_size)
            self.variance_estimator=Variance_estimator(latent_dim, hidden_dim, data_shape,drop_prob=drop_prob)
            
        def forward(self,x):
            mu=self.encoder_base(x)
            sigma=self.variance_estimator(mu)
            return mu, sigma
        
    class Decoder(nn.Module):

        def __init__(self, latent_dim, hidden_dim, batch_size,data_shape,pca,drop_prob,barycenter):
            super().__init__()
            self.decoder_base=Decoder_base(latent_dim=latent_dim, hidden_dim=hidden_dim, data_shape=data_shape,pca=pca,batch_size=batch_size,drop_prob=drop_prob,barycenter=barycenter)

        def forward(self,x):
            return self.decoder_base(x)
    

    def __init__(self,data_shape,pca,latent_dim,batch_size,drop_prob,barycenter,hidden_dim: int= 300, beta=0.000001,**kwargs):
        super().__init__()
        self.pca=pca
        self.barycenter=barycenter
        self.drop_prob=drop_prob
        self.latent_dim=latent_dim
        self.batch_size=batch_size
        self.beta=beta
        self.hidden_dim=hidden_dim
        self.log_scale=nn.Parameter(torch.tensor([0.0]))
        self.data_shape = data_shape
        self.encoder = self.Encoder(data_shape=self.data_shape, latent_dim=self.latent_dim,hidden_dim=self.hidden_dim,pca=self.pca,drop_prob=self.drop_prob,batch_size=self.batch_size)
        self.decoder = self.Decoder(latent_dim=self.latent_dim,hidden_dim=self.hidden_dim ,data_shape=self.data_shape,drop_prob=self.drop_prob,pca=self.pca,batch_size=batch_size,barycenter=self.barycenter)
        self.automatic_optimization=False
    
    def training_step(self, batch, batch_idx):
        opt=self.optimizers()
        x=batch
        mu_1,sigma_1 = self.encoder(x)
        q_1 = torch.distributions.Normal(mu_1.reshape(self.batch_size,-1), sigma_1.reshape(self.batch_size,-1))
        standard_1=torch.distributions.Normal(torch.zeros_like(mu_1.reshape(self.batch_size,-1)), torch.ones_like(sigma_1.reshape(self.batch_size,-1)))
        z1_sampled = q_1.rsample()
        x_hat = self.decoder(z1_sampled)
        p_1=torch.distributions.Normal(x_hat.reshape(self.batch_size,-1),torch.exp(self.log_scale))
        reconstruction=p_1.log_prob(x.reshape(self.batch_size,-1)).mean(dim=1)
        reg=torch.distributions.kl_divergence(q_1, standard_1).mean(dim=1)
        elbo=(reconstruction-self.beta*reg).mean(dim=0)
        elbo_loss=-elbo
        opt.zero_grad()
        self.manual_backward(elbo_loss)
        self.clip_gradients(opt, gradient_clip_val=0.1)
        opt.step()
        return elbo_loss
    
    
    def get_latent(self,data):
        return self.encoder.forward(data)[0]

    
    def validation_step(self, batch, batch_idx):
        x=batch
        mu,sigma = self.encoder(x)
        q_1 = torch.distributions.Normal(mu,sigma)
        standard_1=torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(sigma))
        z1_sampled = q_1.rsample()
        x_hat = self.decoder(z1_sampled)
        x_hat=x_hat.reshape(x.shape)

        loss=L2_loss(x_hat, x)
        reg=torch.distributions.kl_divergence(q_1, standard_1).mean()
        self.log("val_vae_loss", loss)
        return loss+reg

    def test_step(self, batch, batch_idx):
        x=batch
        mu,sigma = self.encoder(x)
        q_1 = torch.distributions.Normal(mu,sigma)
        z1_sampled = q_1.rsample()
        x_hat = self.decoder(z1_sampled)
        x_hat=x_hat.reshape(x.shape)
        loss = torch.linalg.norm(x-x_hat)/torch.linalg.norm(x)
        self.log("test_vae_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
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
        temp_interior=self.decoder(z)
        return temp_interior,z

