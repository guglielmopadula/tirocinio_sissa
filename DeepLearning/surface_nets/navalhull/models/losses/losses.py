
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
import torch
import gpytorch

kernel=gpytorch.kernels.RBFKernel()

def L2_loss(x_hat, x):
    x_hat=x_hat.reshape(x_hat.shape[0],-1)
    x=x.reshape(x.shape[0],-1)
    loss=F.mse_loss(x.reshape(-1), x_hat.reshape(-1), reduction="none")
    loss=loss.mean()
    return loss
 
def LI_loss(x_hat, x):
    x_hat=x_hat.reshape(x_hat.shape[0],-1)
    x=x.reshape(x.shape[0],-1)
    loss=torch.linalg.vector_norm(x-x_hat,dim=1)
    loss=loss.sum()
    return loss


def relativemmd(X,Y):
    X=X.reshape(X.shape[0],-1)
    Y=Y.reshape(Y.shape[0],-1)
    return np.sqrt((1/(len(X)**2)*np.sum(pairwise_kernels(X, X, metric='laplacian'))+1/(len(Y)**2)*np.sum(pairwise_kernels(Y, Y, metric='laplacian'))-2/(len(X)*len(Y))*np.sum(pairwise_kernels(X, Y, metric='laplacian'))))/(np.sqrt(1/(len(X)**2)*np.sum(pairwise_kernels(X, X, metric='laplacian')))+np.sqrt(1/(len(Y)**2)*np.sum(pairwise_kernels(Y, Y, metric='laplacian'))))
    
def mmd(X,Y):
    X=X.reshape(X.shape[0],-1)
    Y=Y.reshape(Y.shape[0],-1)
    return np.sqrt((1/(len(X)**2)*np.sum(pairwise_kernels(X, X, metric='laplacian'))+1/(len(Y)**2)*np.sum(pairwise_kernels(Y, Y, metric='laplacian'))-2/(len(X)*len(Y))*np.sum(pairwise_kernels(X, Y, metric='laplacian'))))

def CE_loss(x_hat,x):
    loss=F.binary_cross_entropy(x_hat,x)
    loss=loss.mean()
    return loss


def compute_kernel(x, y):
    alpha=torch.tensor([0.01,0.1,1,10,100],device=x.device)
    n=len(x)
    m=len(y)
    norms_1 = torch.sum(x**2, dim=1, keepdim=True)
    norms_2 = torch.sum(y**2, dim=1, keepdim=True)
    norms = (norms_1.expand(n, m) + norms_2.transpose(0, 1).expand(n, m))
    distances_squared = norms - 2 * x.mm(y.t())
    distances_repeated=alpha.unsqueeze(1).unsqueeze(1).repeat(1,m,n)*distances_squared.unsqueeze(0).repeat(5,1,1)
    return torch.sum(torch.exp(-torch.sqrt(torch.abs(distances_repeated))),axis=0)


def torch_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2*torch.mean(xy_kernel)


def torch_mmd(x, y):
    x_kernel=compute_kernel(x,x)
    y_kernel=compute_kernel(x,x)
    xy_kernel=compute_kernel(x,y)

    #y_kernel = kernel.covar_dist(y, y)
    #xy_kernel = kernel.covar_dist(x, y)
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2*torch.mean(xy_kernel)