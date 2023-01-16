
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels


def L2_loss(x_hat, x):
    loss=F.mse_loss(x.reshape(-1), x_hat.reshape(-1), reduction="none")
    loss=loss.mean()
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