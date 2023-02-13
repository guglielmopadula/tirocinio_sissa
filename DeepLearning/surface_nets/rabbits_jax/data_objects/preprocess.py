import numpy as np

def unique_qr(A):
    Q, R = np.linalg.qr(A)
    signs = 2 * (np.diag(R) >= 0) - 1
    Q = Q * signs[np.newaxis, :]
    R = R * signs[:, np.newaxis]
    return Q, R
class PCA():
    def __init__(self,reduced_dim):
        self._reduced_dim=reduced_dim
        
    def fit(self,matrix):
        self._n=matrix.shape[0]
        self._p=matrix.shape[1]
        mean=np.mean(matrix,axis=0)
        self._mean_matrix=np.matmul(np.ones((self._n,1)),mean.reshape(1,self._p))
        X=matrix-self._mean_matrix
        Cov=np.matmul(X.t(),X)/self._n
        self._V,S,_=np.linalg.svd(Cov)
        self._V=self._V[:,:self._reduced_dim]
        
    def transform(self,matrix):
        return np.matmul(matrix-self._mean_matrix[:matrix.shape[0],:],self._V)
    
    def inverse_transform(self,matrix):
        return np.matmul(matrix,self._V.t())+self._mean_matrix[:matrix.shape[0],:]
    
class VolumePreservingEmbedding():
    def __init__(self,A,b):
        self.A=A
        self.b=b
        self.sol=np.matmul(A.T,np.linalg.solve(np.matmul(A,A.T),b))
        Q,R=unique_qr(A)
        self.null_space=Q[:,len(A)]

    def transform(self,x):
        x=x-self.sol
        x_red=np.dot(x,self.null_space.T,axis=0)