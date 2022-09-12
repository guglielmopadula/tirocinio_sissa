#!/usr/bin/env python
# coding: utf-8

import numpy as np
from stl import mesh
import stl
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from ordered_set import OrderedSet
import pyro.poutine as poutine
import torch.distributions.constraints as constraints
device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_cuda=True if torch.cuda.is_available() else False
torch.manual_seed(0)
import math
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def getinfo(stl):
    your_mesh = mesh.Mesh.from_file(stl)
    myList = list(OrderedSet(tuple(map(tuple,your_mesh.vectors.reshape(np.prod(your_mesh.vectors.shape)//3,3)))))
    array=your_mesh.vectors
    topo=np.zeros((np.prod(your_mesh.vectors.shape)//9,3))
    for i in range(np.prod(your_mesh.vectors.shape)//9):
        for j in range(3):
            topo[i,j]=myList.index(tuple(array[i,j].tolist()))
    return torch.tensor(myList),torch.tensor(topo, dtype=torch.int64)
    
def applytopology(V,M):
    Q=torch.zeros((M.shape[0],3,3),device=device)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            Q[i,j]=V[M[i,j].item()]
    return Q

def rescale(x):
    temp=x.shape
    x=x.reshape(x.shape[0],-1,3)
    return (x/((torch.abs(torch.det(x[:,M])).sum(1).reshape(x.shape[0],1).expand(x.shape[0],x.numel()//x.shape[0]).reshape(x.shape)/6)**1/3)).reshape(temp)

def calcvolume(x):
    return x[M].det().abs().sum()/6

data=[]
for i in range(100):
    if i%100==0:
        print(i)
    points,M=getinfo("bulbo_{}.stl".format(i))
    if device!='cpu':
        points=points.to(device)
    data.append(points)
    
    
K=torch.numel(points)
if device!='cpu':
    M=M.to(device)
    
datatrain=data[:len(data)//2]
datatest=data[len(data)//2:]
datatraintorch=torch.zeros(len(datatrain),datatrain[0].shape[0],datatrain[0].shape[1],dtype=datatrain[0].dtype, device=device)
datatesttorch=torch.zeros(len(datatest),datatest[0].shape[0],datatest[0].shape[1],dtype=datatest[0].dtype, device=device)
for i in range(len(datatrain)):
    datatraintorch[i:]=datatrain[i]
for i in range(len(datatest)):
    datatesttorch[i:]=datatest[i]

class VolumeNormalizer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        temp=x.shape
        x=x.reshape(x.shape[0],-1,3)
        x=x/((x[:,M].det().abs().sum(1)/6)**(1/3)).reshape(-1,1).expand(x.shape[0],x.numel()//x.shape[0]).reshape(x.shape[0],-1,3)
        return x.reshape(temp)
    
    def forward_single(self,x):
        temp=x.shape
        x=x.reshape(1,-1,3)
        x=x/((x[:,M].det().abs().sum(1)/6)**(1/3)).reshape(-1,1).expand(x.shape[0],x.numel()//x.shape[0]).reshape(1,-1,3)
        return x.reshape(temp)

        

class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, K)
        self.fc5=VolumeNormalizer()
        self.relu = nn.ReLU()

    def forward(self, z):
        result=self.fc4(self.relu(self.relu(self.fc3(self.relu(self.fc2(self.relu(self.fc1(z))))))))       
        #result=self.fc5(result)
        return result
    
    def sample(self,z):
        result=self.fc4(self.relu(self.relu(self.fc3(self.relu(self.fc2(self.relu(self.fc1(z))))))))
        result=self.fc5.forward_single(result)
        return result

    

class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(K,hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, hidden_dim)
        self.fc31 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        self.fc41=nn.Tanh()
        self.fc32 = nn.Sigmoid()
        self.batch=nn.BatchNorm1d(1)



    def forward(self, x):
        x=x.reshape(-1,K)
        hidden=self.fc1(x)
        mu=self.fc31(self.fc21(hidden))
        mu=self.batch(mu)
        mu=mu/torch.linalg.norm(mu)*torch.tanh(torch.linalg.norm(mu))*(2/math.pi)
        sigma=torch.exp(self.fc32(self.fc22(hidden)))-0.9
        return mu,sigma

        
class VAE(nn.Module):
    def __init__(self, z_dim=1, hidden_dim=300, use_cuda=False):
        super().__init__()
        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)
        if use_cuda:
            self.cuda()
        self.use_cuda=use_cuda
        self.z_dim = z_dim
        
    @poutine.scale(scale=1.0/datatraintorch.shape[0])
    def model(self,x):
        pyro.module("decoder", self.decoder)
        mu = pyro.param("mu",torch.zeros( self.z_dim, dtype=x.dtype, device=x.device))
        sigma = pyro.param("sigma",torch.ones(self.z_dim, dtype=x.dtype, device=x.device),constraint=constraints.positive)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z_scale=torch.cat(x.shape[0]*[mu], 0).reshape(-1,self.z_dim)
            z_loc=torch.cat(x.shape[0]*[sigma], 0).reshape(-1,self.z_dim)
            z = pyro.sample("latent", dist.Normal(z_scale, z_loc).to_event(1))
            # decode the latent code z
            x_hat = self.decoder.forward(z)
            # score against actual images (with relaxed Bernoulli values)
            pyro.sample(
                "obs",
                dist.Normal(x_hat, (1e-7)*torch.ones(x_hat.shape, dtype=x.dtype, device=x.device), validate_args=False).to_event(1),
                obs=x.reshape(-1, K),
            )
            # return the loc so we can visualize it later
            return x_hat

    @poutine.scale(scale=1.0/datatraintorch.shape[0])   
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    
    def apply_vae_verbose(self,x):
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        print("scale is",z_scale,"mean is", z_loc)
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        return loc_img
    
    def apply_vae_point(self,x):
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        return loc_img

    def apply_vae_mesh(self,x):
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        loc_img=applytopology(loc_img.reshape(K//3,3),M)
        return loc_img


    def sample_mesh(self):
        mu = pyro.param("mu")
        sigma = pyro.param("sigma") 
        z = pyro.sample("latent", dist.Normal(mu, sigma))
        a=self.decoder.sample(z)
        mesh=applytopology(a.reshape(K//3,3),M)
        return mesh
    

    
def train(vae,datatraintorch,datatesttorch,epochs=5000):
    pyro.clear_param_store()
    elbotrain=[]
    elbotest=[]
    errortest=[]
    adam_args = {"lr": 0.001}
    optimizer = Adam(adam_args)
    elbo = Trace_ELBO()
    svi = SVI(vae.model, vae.guide, optimizer, loss=elbo)
    for epoch in range(epochs):
        print(epoch)
        if epoch%1000==0:
            print(epoch)
        elbotest.append(svi.evaluate_loss(datatesttorch))
        temp=(1/(K*len(datatesttorch)))*(((vae.apply_vae_point(datatesttorch)-datatesttorch.reshape(len(datatesttorch),K))**2).sum())
        print(temp)
        errortest.append(temp.clone().detach().cpu())
        elbotrain.append(svi.step(datatraintorch))
    return elbotrain,elbotest,errortest



vae = VAE(use_cuda=use_cuda)



#vae.load_state_dict(torch.load("cube.pt"))
elbotrain,elbotest,errortest = train(vae,datatraintorch, datatesttorch)


fig, axs = plt.subplots(2)

axs[0].plot([i for i in range(len(elbotrain))],elbotrain)
axs[0].plot([i for i in range(len(elbotest))],elbotest)
axs[1].plot([i for i in range(len(errortest))],errortest)


lst=vae.encoder(datatesttorch)[0].clone().detach().cpu().numpy()
    
lst=np.array(lst)

'''
def calculate_WSS(points, kmax):
  sse = []
  for k in range(1, kmax+1):
    kmeans = KMeans(n_clusters = k).fit(points)
    centroids = kmeans.cluster_centers_
    pred_clusters = kmeans.predict(points)
    curr_sse = 0
    
    # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
    for i in range(len(points)):
      curr_center = centroids[pred_clusters[i]]
      curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2
      
    sse.append(curr_sse)
  return sse

wss=calculate_WSS(np.array(lst),20)


sil = []
kmax = 20

# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
for k in range(2, kmax+1):
  kmeans = KMeans(n_clusters=k).fit(lst)
  labels = kmeans.labels_
  sil.append(silhouette_score(lst, labels, metric='euclidean'))

k=5
kmeans = KMeans(n_clusters=k, random_state=0).fit(lst)
index=np.argmax(np.bincount(kmeans.predict(lst)))
lst=np.array(lst)
realspace=lst[kmeans.predict(lst)==0]
mean=np.mean(realspace,axis=1)
np.linalg.norm((np.mean(realspace,axis=1).reshape(-1,1)-realspace),axis=1)
'''
temp = vae.sample_mesh()
data = np.zeros(len(temp), dtype=mesh.Mesh.dtype)
data['vectors'] = temp.cpu().detach().numpy().copy()
mymesh = mesh.Mesh(data.copy())
mymesh.save('test.stl', mode=stl.Mode.ASCII)