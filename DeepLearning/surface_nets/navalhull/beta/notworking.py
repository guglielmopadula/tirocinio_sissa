#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 17:02:36 2023

@author: cyberguli
"""


'''
class LST(nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        self.lin=torch.nn.utils.parametrizations.spectral_norm(nn.Linear(in_features, out_features))
        self.tanh=nn.Tanh()
        self.dropout=nn.Dropout(DROP_PROB)

    
    def forward(self,x):
        return self.dropout(self.tanh(self.lin(x)))



class Discriminator_base(nn.Module):
    def __init__(self, hidden_dim,data_shape,pca_1,pca_2):
        super().__init__()
        self.data_shape=data_shape
        self.pca_1=pca_1
        self.pca_2=pca_2
        self.fc_interior_1 = LST(int(np.prod(self.data_shape[0])), hidden_dim)
        self.fc_interior_2 = LST(hidden_dim, hidden_dim)
        self.fc_interior_3 = LST(hidden_dim, hidden_dim)
        self.fc_interior_4 = LST(hidden_dim, hidden_dim)
        self.fc_interior_5 = LST(hidden_dim, hidden_dim)
        self.fc_interior_6 = LST(hidden_dim, hidden_dim)
        self.fc_interior_7 = torch.nn.utils.parametrizations.spectral_norm(nn.Linear(hidden_dim, 1))
        self.fc_boundary_1 = LST(int(np.prod(self.data_shape[1])), hidden_dim)
        self.fc_boundary_2 = LST(hidden_dim, hidden_dim)
        self.fc_boundary_3 = LST(hidden_dim, hidden_dim)
        self.fc_boundary_4 = LST(hidden_dim, hidden_dim)
        self.fc_boundary_5 = LST(hidden_dim, hidden_dim)
        self.fc_boundary_6 = LST(hidden_dim, hidden_dim)
        self.fc_boundary_7 = torch.nn.utils.parametrizations.spectral_norm(nn.Linear(hidden_dim, 1))

        
    def forward(self,x,y):
        x=x.reshape(x.size(0),-1)
        x=self.pca_1.transform(x)
        y=y.reshape(y.size(0),-1)
        y=self.pca_2.transform(y)
        out_1=self.fc_interior_7(self.fc_interior_6(self.fc_interior_5(self.fc_interior_4(self.fc_interior_3(self.fc_interior_2(self.fc_interior_1(x)))))))
        out_2=self.fc_boundary_7(self.fc_boundary_6(self.fc_boundary_5(self.fc_boundary_4(self.fc_boundary_3(self.fc_boundary_2(self.fc_boundary_1(y)))))))
        return out_1+out_2


class WGAN_SP(LightningModule):
    class Generator(nn.Module):
        def __init__(self, latent_dim_1,latent_dim_2, hidden_dim, data_shape,temp_zero,newtriangles_zero,pca_1,pca_2,edge_matrix,vertices_face,cvxpylayer,k,local_indices_1,local_indices_2):
            super().__init__()
            self.decoder_base=Decoder_base(latent_dim_1=latent_dim_1,latent_dim_2=latent_dim_2, hidden_dim=hidden_dim, data_shape=data_shape,local_indices_1=local_indices_1,local_indices_2=local_indices_2,temp_zero=temp_zero,newtriangles_zero=newtriangles_zero,pca_1=pca_1,pca_2=pca_2,edge_matrix=edge_matrix,vertices_face=vertices_face,cvxpylayer=cvxpylayer,k=k)
        def forward(self,x,y):
            return self.decoder_base(x,y)
    class Discriminator(nn.Module):
        def __init__(self, hidden_dim,data_shape,pca_1,pca_2):
            super().__init__()
            self.discriminator=Discriminator_base(hidden_dim=hidden_dim, data_shape=data_shape,pca_1=pca_1,pca_2=pca_2)
        def forward(self,x,y):
            x_hat=self.discriminator(x,y)
            return x_hat

    
    def __init__(self,data_shape,temp_zero,local_indices_1,local_indices_2,newtriangles_zero,pca_1,pca_2,edge_matrix,vertices_face,cvxpylayer,k=SMOOTHING_DEGREE,hidden_dim: int= 300,latent_dim_1: int = LATENT_DIM_1,latent_dim_2: int = LATENT_DIM_1,lr: float = 0.0002,b1: float = 0.5,b2: float = 0.999,batch_size: int = BATCH_SIZE,ae_hyp=0.999999,**kwargs):
        super().__init__()
        #self.save_hyperparameters()
        super().__init__()
        #self.save_hyperparameters()
        self.temp_zero=temp_zero
        self.newtriangles_zero=newtriangles_zero
        self.pca_1=pca_1
        self.local_indices_1=local_indices_1
        self.local_indices_2=local_indices_2
        self.pca_2=pca_2
        self.edge_matrix=edge_matrix
        self.ae_hyp=ae_hyp
        self.k=k
        self.latent_dim_1=latent_dim_1
        self.latent_dim_2=latent_dim_2
        self.hidden_dim=hidden_dim
        self.vertices_face=vertices_face
        self.cvxpylayer=cvxpylayer
        # networks
        self.data_shape = data_shape
        self.generator = self.Generator(latent_dim_1=self.latent_dim_1,latent_dim_2=self.latent_dim_2,hidden_dim=self.hidden_dim ,data_shape=self.data_shape,local_indices_1=self.local_indices_1,local_indices_2=self.local_indices_2,temp_zero=self.temp_zero,newtriangles_zero=self.newtriangles_zero,pca_1=self.pca_1,pca_2=self.pca_2,edge_matrix=self.edge_matrix,vertices_face=self.vertices_face,cvxpylayer=self.cvxpylayer,k=self.k)
        self.discriminator=self.Discriminator(data_shape=self.data_shape,hidden_dim=self.hidden_dim,pca_1=self.pca_1,pca_2=self.pca_2)

    def forward(self):
        z=torch.randn(self.latent_dim)
        x_hat=self.generator(z)
        return x_hat
    
    def ae_loss(self, x_hat, x):
        loss=F.mse_loss(x, x_hat, reduction="none")
        loss=loss.mean()
        return loss
    
    def adversarial_loss(self,y_hat,y):
        return F.binary_cross_entropy(y_hat, y)


    def training_step(self, batch, batch_idx, optimizer_idx ):
        x,y=batch
        standard_1 = torch.distributions.Normal(torch.zeros(len(x),self.latent_dim_1), torch.ones(len(x),self.latent_dim_1))
        standard_2 = torch.distributions.Normal(torch.zeros(len(y),self.latent_dim_2), torch.ones(len(y),self.latent_dim_2))
        z1_sampled = standard_1.rsample()
        z2_sampled = standard_2.rsample()
        x_hat,y_hat = self.generator(z1_sampled,z2_sampled)
        x_hat=x_hat.reshape(x.shape)
        y_hat=y_hat.reshape(y.shape)
        
        if optimizer_idx==0:
            g_loss = -self.discriminator(x_hat,y_hat).mean()
            if LOGGING:
                self.log("wgan_gen_train_loss", g_loss,on_step=False, on_epoch=True, prog_bar=True)
            return g_loss
        
        if optimizer_idx==1:
            real_loss = -self.discriminator(x,y).mean()
            fake_loss = self.discriminator(x_hat,y_hat).mean()
            tot_loss= (real_loss+fake_loss)/2
            if LOGGING:
                self.log("wgan_disc_train_loss", tot_loss,on_step=False, on_epoch=True, prog_bar=True)
            return tot_loss
            
            
    
    def validation_step(self, batch, batch_idx):
        x,y=batch
        standard_1 = torch.distributions.Normal(torch.zeros(len(x),self.latent_dim_1), torch.ones(len(x),self.latent_dim_1))
        standard_2 = torch.distributions.Normal(torch.zeros(len(y),self.latent_dim_2), torch.ones(len(y),self.latent_dim_2))
        z1_sampled = standard_1.rsample()
        z2_sampled = standard_2.rsample()
        x_hat,y_hat = self.generator(z1_sampled,z2_sampled)
        x_hat=x_hat.reshape(x.shape)
        y_hat=y_hat.reshape(y.shape)        
        real_loss = -self.discriminator(x,y).mean()
        fake_loss = self.discriminator(x_hat,y_hat).mean()
        tot_loss= (real_loss+fake_loss)/2
        if LOGGING:
            self.log("wgan_val_loss", tot_loss, on_step=False, on_epoch=True, prog_bar=True)
        return tot_loss
        
    
    def test_step(self, batch, batch_idx):
        x,y=batch
        standard_1 = torch.distributions.Normal(torch.zeros(len(x),self.latent_dim_1), torch.ones(len(x),self.latent_dim_1))
        standard_2 = torch.distributions.Normal(torch.zeros(len(y),self.latent_dim_2), torch.ones(len(y),self.latent_dim_2))
        z1_sampled = standard_1.rsample()
        z2_sampled = standard_2.rsample()
        x_hat,y_hat = self.generator(z1_sampled,z2_sampled)
        x_hat=x_hat.reshape(x.shape)
        y_hat=y_hat.reshape(y.shape)        
        loss=0.5*torch.min(torch.norm(x_hat-x,dim=(1,2))/torch.norm(x,dim=(1,2)))+0.5*torch.min(torch.norm(y_hat-y,dim=(1,2))/torch.norm(x,dim=(1,2)))
        if LOGGING:
            self.log("wgan_test_loss", loss,on_step=False, on_epoch=True, prog_bar=True)
        return loss

        
    def configure_optimizers(self):#
        optimizer_g = torch.optim.SGD(self.generator.parameters(), lr=0.0001)
        optimizer_d = torch.optim.SGD(self.discriminator.parameters(), lr=0.0004) 
        return [optimizer_g,optimizer_d], []

    def sample_mesh(self):
        mean_1=torch.zeros(1,self.latent_dim_1)
        mean_2=torch.zeros(1,self.latent_dim_2)
        var_1=torch.ones(1,self.latent_dim_1)
        var_2=torch.ones(1,self.latent_dim_2)
        z = torch.sqrt(var_1)*torch.randn(1,self.latent_dim_1)+mean_1
        w = torch.sqrt(var_2)*torch.randn(1,self.latent_dim_2)+mean_2
        temp_interior,temp_boundary=self.generator(z,w)
        return temp_interior,temp_boundary
'''



'''
class LaplaceData(LightningModule):
    def __init__(self,data_shape,temp_zero,local_indices,M1,M2,M3,pca,edge_matrix,k=SMOOTHING_DEGREE,hidden_dim: int= 300,latent_dim: int = LATENT_DIM,lr: float = 0.0002,b1: float = 0.5,b2: float = 0.999,batch_size: int = BATCH_SIZE, ae_hyp=0.999,**kwargs):
        super().__init__()
        self.data=torch.zeros(NUM_LAPL,1328,3)
        for i in range(NUM_LAPL):
            self.data[i]=getinfo("hull_negative_{}.stl".format(i),False)[0]
        self.lin=nn.Linear(1, 1)
        self.c=torch.tensor([5.])
        self.i=0
        
        
    def forward(self, x):
        return 0
    
    def training_step(self, batch, batch_idx):
        return torch.linalg.norm(self.c-self.lin(self.c))    
                
    
    def validation_step(self, batch, batch_idx):
        return torch.linalg.norm(self.c-self.lin(self.c))    
        
    def test_step(self, batch, batch_idx):
        return torch.linalg.norm(self.c-self.lin(self.c))    
    
    def configure_optimizers(self): #0.039,.0.2470, 0.2747
        optimizer=torch.optim.AdamW(self.lin.parameters(), lr=0.02)#0.02
        return optimizer

    def sample_mesh(self,mean=None,var=None):
        t=self.data[self.i]
        self.i=(self.i+1)%NUM_LAPL
        return t
    
'''    


'''
class VAEWGAN(LightningModule):
    
    
    class Encoder(nn.Module):
        def __init__(self, latent_dim, hidden_dim,data_shape,pca):
            super().__init__()
            self.encoder_base=Encoder_base(latent_dim, hidden_dim, data_shape,pca)
            self.variance_estimator=Variance_estimator(latent_dim, hidden_dim, data_shape)
            
        def forward(self,x):
            mu=self.encoder_base(x)
            sigma=self.variance_estimator(mu)
            return mu,sigma

    class Decoder(nn.Module):
        def __init__(self, latent_dim, hidden_dim, data_shape,temp_zero,newtriangles_zero,pca,edge_matrix,vertices_face,cvxpylayer,k):
            super().__init__()
            self.decoder_base=Decoder_base(latent_dim, hidden_dim, data_shape,temp_zero,newtriangles_zero,pca,edge_matrix,vertices_face,cvxpylayer,k)
            
        def forward(self,x):
            return self.decoder_base(x)
    
    
    class Discriminator(nn.Module):
        def __init__(self, latent_dim, hidden_dim,data_shape,pca):
            super().__init__()
            self.discriminator_base=Discriminator_base(latent_dim, hidden_dim, data_shape,pca)
             
        def forward(self,x):
            x_hat=self.discriminator_base(x)
            return x_hat

    
    def __init__(self,data_shape,temp_zero,newtriangles_zero,pca,edge_matrix,vertices_face,cvxpylayer,k=SMOOTHING_DEGREE,hidden_dim: int= 300,latent_dim: int = LATENT_DIM,lr: float = 0.0002,b1: float = 0.5,b2: float = 0.999,batch_size: int = BATCH_SIZE, ae_hyp=0.999,**kwargs):
        super().__init__()
        #self.save_hyperparameters()
        self.temp_zero=temp_zero
        self.newtriangles_zero=newtriangles_zero
        self.pca=pca
        self.edge_matrix=edge_matrix
        self.k=k
        self.latent_dim=latent_dim
        self.hidden_dim=hidden_dim
        self.vertices_face=vertices_face
        self.cvxpylayer=cvxpylayer
        self.data_shape = data_shape
        self.decoder = self.Decoder(latent_dim=self.latent_dim,hidden_dim=self.hidden_dim ,data_shape=self.data_shape,temp_zero=self.temp_zero,newtriangles_zero=self.newtriangles_zero,pca=self.pca,edge_matrix=self.edge_matrix,vertices_face=self.vertices_face,cvxpylayer=self.cvxpylayer,k=self.k)
        self.encoder = self.Encoder(data_shape=self.data_shape, latent_dim=self.latent_dim ,hidden_dim=self.hidden_dim,pca=self.pca)
        self.discriminator=self.Discriminator(hidden_dim=self.hidden_dim,data_shape=self.data_shape,latent_dim=self.latent_dim,pca=self.pca)
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        self.ae_hyp=ae_hyp
        
    def forward(self, x):
        z=self.encoder(x)
        x_hat=self.decoder(z)
        return x_hat.reshape(x.shape)
    
    def ae_loss(self, x_hat, x):
        loss=F.mse_loss(x, x_hat, reduction="none")
        loss=loss.mean()
        return loss
    
    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum()

    def kl_divergence(self, z, mu, std):
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)
        kl_div = (log_qzx - log_pz)
        kl_div = kl_div.sum(-1)
        return kl_div

    
    
    def adversarial_loss(self,y_hat,y):
        return F.binary_cross_entropy(y_hat, y)


    def training_step(self, batch, batch_idx, optimizer_idx ):
        mu,sigma=self.encoder(batch)
        q = torch.distributions.Normal(mu, sigma)
        z_sampled = q.rsample().type_as(batch)
        batch=batch.reshape(batch.shape[0],-1)
        disl=self.discriminator(batch)
        batch_hat = self.decoder(z_sampled).reshape(batch.shape)
        disl_hat=self.discriminator(batch_hat)
        z_p=torch.randn(len(batch), self.latent_dim).type_as(batch)
        batch_p=self.decoder(z_p)
        ldisc = -self.gaussian_likelihood(disl_hat, self.log_scale, disl).mean()

        lprior = self.kl_divergence(z_sampled, mu, sigma).mean()
        lgan=(torch.log(self.discriminator(batch))-0.5*torch.log(self.discriminator(batch_hat))-0.5*torch.log(self.discriminator(batch_p))).mean()
        
        if optimizer_idx==0:
            loss=lprior+ldisc
            if LOGGING:
                self.log("train_encoder_loss", loss)
            return loss.mean()
        

        if optimizer_idx==1:    
            loss = -self.ae_hyp*ldisc-(1-self.ae_hyp)*lgan
            if LOGGING:
                self.log("train_decoder_loss", loss)
            return loss
        
        if optimizer_idx==2:
            if LOGGING:
                self.log("train_discriminator_loss", lgan)
            return lgan
            
                
    
    def validation_step(self, batch, batch_idx):
        mu,sigma = self.encoder(batch)
        batch_hat=self.decoder(mu).reshape(batch.shape)
        if LOGGING:
            self.log("val_vaewgan_loss", self.ae_loss(batch,batch_hat))
        return self.ae_loss(batch,batch_hat)

        
    def test_step(self, batch, batch_idx):
        mu,sigma = self.encoder(batch)
        batch_hat=self.decoder(mu).reshape(batch.shape)
        if LOGGING:
            self.log("test_vaewgan_loss", self.ae_loss(batch,batch_hat))
        return self.ae_loss(batch,batch_hat)
    
        
    def get_latent(self,data):
        return self.encoder.forward(data)[0]


    

    def configure_optimizers(self): #0.039,.0.2470, 0.2747
        optimizer_enc=torch.optim.AdamW(self.encoder.parameters(), lr=0.05, weight_decay=0.1)#0.02
        optimizer_dec = torch.optim.AdamW(self.decoder.parameters(), lr=0.05,weight_decay=0.1) #0.02
        optimizer_disc = torch.optim.AdamW(self.discriminator.parameters(), lr=0.1, weight_decay=0.1) #0.050
        return [optimizer_enc,optimizer_dec,optimizer_disc], []

    def sample_mesh(self,mean=None,var=None):
        if mean==None:
            mean=torch.zeros(1,self.latent_dim)
        if var==None:
            var=torch.ones(1,self.latent_dim)
        z = torch.sqrt(var)*torch.randn(1,self.latent_dim)+mean
        temp=self.decoder(z)
        return temp
'''


