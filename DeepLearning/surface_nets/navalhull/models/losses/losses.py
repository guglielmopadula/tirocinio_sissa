
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
import torch

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
    x_size = x.shape[0]
    y_size = y.shape[0]
    dim = x.shape[1]

    tiled_x = x.view(x_size,1,dim).repeat(1, y_size,1)
    tiled_y = y.view(1,y_size,dim).repeat(x_size, 1,1)

    return torch.exp(-torch.mean((tiled_x - tiled_y)**2,dim=2)/dim*1.0)


def torch_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2*torch.mean(xy_kernel)