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
        def __init__(self, latent_dim, hidden_dim,data_shape,pca,drop_prob,batch_size):
            super().__init__()
            self.encoder_base=Encoder_base(latent_dim=latent_dim, hidden_dim=hidden_dim, data_shape=data_shape,pca=pca,drop_prob=drop_prob,batch_size=batch_size)
            
        def forward(self,x):
            z=self.encoder_base(x)
            return z
        
    class Decoder(nn.Module):

        def __init__(self, latent_dim, hidden_dim, batch_size,data_shape,pca,drop_prob,barycenter):
            super().__init__()
            self.decoder_base=Decoder_base(latent_dim=latent_dim, hidden_dim=hidden_dim, data_shape=data_shape,pca=pca,batch_size=batch_size,drop_prob=drop_prob,barycenter=barycenter)

        def forward(self,x):
            return self.decoder_base(x)
    
    class Discriminator(nn.Module):
        def __init__(self, latent_dim, hidden_dim,data_shape,drop_prob):
            super().__init__()
            self.discriminator=Latent_Discriminator_base(latent_dim=latent_dim, data_shape=data_shape,hidden_dim=hidden_dim ,drop_prob=drop_prob)
             
        def forward(self,x):
            x_hat=self.discriminator(x)
            return x_hat



    def __init__(self,data_shape,pca,latent_dim,batch_size,drop_prob,barycenter,hidden_dim: int= 300, ae_hyp=0.999,**kwargs):
        super().__init__()
        self.pca=pca
        self.ae_hyp=ae_hyp
        self.barycenter=barycenter
        self.drop_prob=drop_prob
        self.latent_dim=latent_dim
        self.batch_size=batch_size
        self.hidden_dim=hidden_dim
        self.data_shape = data_shape
        self.encoder = self.Encoder(data_shape=self.data_shape, latent_dim=self.latent_dim,hidden_dim=self.hidden_dim,pca=self.pca,drop_prob=self.drop_prob,batch_size=self.batch_size)
        self.decoder = self.Decoder(latent_dim=self.latent_dim,hidden_dim=self.hidden_dim ,data_shape=self.data_shape,drop_prob=self.drop_prob,pca=self.pca,batch_size=batch_size,barycenter=self.barycenter)
        self.discriminator=self.Discriminator(data_shape=self.data_shape, latent_dim=self.latent_dim,hidden_dim=self.hidden_dim,drop_prob=drop_prob)
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        x=batch
        z_enc=self.encoder(x)
        z_1=torch.randn(len(x), self.latent_dim).type_as(x)
        x_disc_e=self.discriminator(z_enc)
        x_disc=self.discriminator(z_1)
        x_hat=self.decoder(z_enc)
        x_hat=x_hat.reshape(x.shape)
        ae_opt, d_opt = self.optimizers()
        ae_loss = self.ae_hyp*L2_loss(x_hat ,x)+(1-self.ae_hyp)*(CE_loss(x_disc_e,torch.ones_like(x_disc_e)).mean())
        ae_opt.zero_grad()
        self.manual_backward(ae_loss)
        self.clip_gradients(ae_opt, gradient_clip_val=0.1)
        ae_opt.step()        
        real_loss = CE_loss(x_disc,torch.ones_like(x_disc)).mean()
        fake_loss = CE_loss(x_disc_e.detach(),torch.zeros_like(x_disc_e)).mean()
        tot_loss= (real_loss+fake_loss)/2
        d_opt.zero_grad()
        self.manual_backward(tot_loss)
        self.clip_gradients(d_opt, gradient_clip_val=0.1)
        d_opt.step()        
        return tot_loss
            
        
        
    def get_latent(self,data):
        return self.encoder.forward(data)
    
    def test_step(self, batch, batch_idx):
        x=batch
        z=self.encoder(x)
        x_hat=self.decoder(z)
        x_hat=x_hat.reshape(x.shape)
        loss = torch.linalg.norm(x-x_hat)/torch.linalg.norm(x)
        self.log("test_aae_loss", loss)
        return loss


    def configure_optimizers(self):
        optimizer_ae = torch.optim.AdamW(itertools.chain(self.encoder.parameters(), self.decoder.parameters()), lr=1e-7)
        optimizer_disc = torch.optim.AdamW(self.discriminator.parameters(), lr=1e-3)
        return [optimizer_ae,optimizer_disc], []
    
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

