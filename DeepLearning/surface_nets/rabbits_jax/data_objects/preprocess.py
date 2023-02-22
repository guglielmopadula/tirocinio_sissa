import numpy as np
import meshio
import scipy
import time
import skdim
from tqdm import tqdm
import sklearn
import pickle
import jax.numpy as jnp
from jax import jit
from jax import config
import jax
import warnings
from functools import partial
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

def rref(A):
    M=A
    l=[]
    lead = 0
    rowCount = M.shape[0]
    columnCount = M.shape[1]
    for r in range(rowCount):
        if lead >= columnCount:
            return M
        i = r
        while M[i,lead] == 0:
            i += 1
            if i == rowCount:
                i = r
                lead += 1
                if columnCount == lead:
                    return M
        M[i],M[r] = M[r],M[i]
        M[r] = M[r]/M[r,lead]
        for i in range(rowCount):
            if i != r:
                M[i] = M[i]-M[r]*M[i,lead] 
        l.append(lead)
        lead += 1

    return M,l

def getinfo(stl):
    mesh=meshio.read(stl)
    points=mesh.points
    barycenter=np.mean(points,axis=0)
    return points,barycenter


class PCA():
    def __init__(self,reduced=None):
        self.reduced=reduced
    def fit(self, data):
        self.mean=jnp.mean(data,axis=0)
        self.nsamples=len(data)
        data=data-self.mean

        M=jnp.einsum("ik,kj->ij", data, data.T, precision=jax.lax.Precision.HIGH)/self.nsamples
        w,V=jnp.linalg.eigh(M)
        w=w[::-1]
        V=V[:,::-1]
        self.explained_variance_ratio_=w/jnp.sum(w)
        self.U=None
        if self.reduced==None:
            self.reduced=len(V)
        else:
            self.reduced=self.reduced

            self.U=jnp.sqrt(1/(self.nsamples*w[:self.reduced]))*jnp.einsum("ik,kj->ij", data.T, V[:,:self.reduced], precision=jax.lax.Precision.HIGH) 
    
    def transform(self,data):
        return _transform(data,self.mean,self.U)


    def inverse_transform(self,data):
        return _inverse_transform(data,self.mean,self.U)


@jit
def _transform(data,mean,U):
    data=data-mean
    return jnp.einsum("ik,kj->ij", data, U, precision=jax.lax.Precision.HIGH)

@jit
def _inverse_transform(data,mean,U):
    return jnp.einsum("ik,kj->ij", data, U.T, precision=jax.lax.Precision.HIGH)+mean


'''
class PCA:
    """PCA in jax.
    Assuming data and intermediate results fit on a local device (RAM or GPU memory).
    No additional setup needed.
    Attributes:
        N: number of principal components. If not given, keeps maximal number of components.
    Methods:
        fit: computing principal vectors.
        transform: calculating principal components for given input.
        inverse_transform: inverse of the transform.
        save: saving the transform.
        load: loading the transform.
        sample: sampling multivariate gaussian distribution of the principal components
            and computing inverse_transform.
    """

    def __init__(self, N=None):
        self.N = N

    def fit(self, data, whiten=False, use_SVD=False):
        """Computing eigenvectors and eigenvalues of the data.
        Args:
            data (np.array): data to fit, of shape `(N_dim, N_samples)`.
            whiten (bool): scaling all dimensions to the unit variance.
            use_SVD (bool): If true, it uses SVD decomposition, which might be
                more stable numerically.
        Returns:
            An instance of itself.
        """
        data = jnp.array(data, dtype=jnp.float32)
        N_dim, N_samples = data.shape
        if self.N is None:
            self.N = min(N_dim, N_samples)

        self.μ = jnp.mean(data, axis=1, keepdims=True, dtype=jnp.float64).astype(
            jnp.float32
        )
        if whiten:
            self.σ = jnp.std(data, axis=1, keepdims=True, dtype=jnp.float64).astype(
                jnp.float32
            )
        else:
            self.σ = jnp.ones((N_dim, 1), dtype=jnp.float32)

        data = (data - self.μ) / self.σ

        if N_dim < N_samples:
            C = jnp.einsum(
                "ik,jk->ij", data, data, precision=jax.lax.Precision.HIGH
            ) / (N_samples - 1)
            try:
                C = C.astype(jnp.float64)
            except:
                warnings.warn("Couldn't use float64 precision for covariance.")
                C = C.astype(jnp.float32)

            if use_SVD:
                self.U, self.eigenvalues, _ = jnp.linalg.svd(
                    C, full_matrices=False, hermitian=True
                )
            else:
                self.eigenvalues, self.U = jnp.linalg.eigh(C)
                self.eigenvalues = self.eigenvalues[::-1]
                self.U = self.U[:, ::-1]

            self.eigenvalues = self.eigenvalues[: self.N]
            if jnp.any(self.eigenvalues < 0):
                warnings.warn("Some eigenvalues are negative.")
            self.λ = jnp.sqrt(self.eigenvalues)
            self.U = self.U[:, : self.N]
        else:
            D = (
                jnp.einsum("ki,kj->ij", data, data, precision=jax.lax.Precision.HIGH)
                / N_dim
            )
            try:
                D = D.astype(jnp.float64)
                print(D.shape)
            except:
                warnings.warn("Couldn't use float64 precision for covariance.")
                D = D.astype(jnp.float32)

            if use_SVD:
                V, self.eigenvalues, _ = jnp.linalg.svd(
                    D, full_matrices=False, hermitian=True
                )
            else:
                self.eigenvalues, V = jnp.linalg.eigh(D)
                self.eigenvalues = self.eigenvalues[::-1]
                V = V[:, ::-1]

            self.eigenvalues = self.eigenvalues[: self.N] * (N_dim / (N_samples - 1))
            if jnp.any(self.eigenvalues < 0):
                warnings.warn("Some eigenvalues are negative.")
            self.λ = jnp.sqrt(self.eigenvalues)
            S_inv = (1 / jnp.sqrt(self.eigenvalues * (N_samples - 1)))[jnp.newaxis, :]
            VS_inv = V[:, : self.N] * S_inv
            self.U = jnp.einsum(
                "ij,jk->ik", data, VS_inv, precision=jax.lax.Precision.HIGH
            ).astype(jnp.float32)

        return self

    def transform(self, X):
        """Transforming X and computing principal components for each sample.
        Args:
            X: data to transform of shape `(N_dim, N_samples)`.
        Returns:
            X_t: transformed data of shape `(N, N_samples)`.
        """
        X = jnp.array(X, dtype=jnp.float32)
        X_t = jnp.einsum("ji,jk->ik", self.U, (X - self.μ) / self.σ)
        return np.array(X_t, dtype=np.float32)

    def inverse_transform(self, X_t):
        """Transforming X_t back to the original space.
        Args:
            X_t: data in principal-components space, of shape `(N, N_samples)`.
        Returns:
            X: transformed data in original space, of shape `(N_dim, N_samples)`.
        """
        X_t = jnp.array(X_t, dtype=jnp.float32)
        X = jnp.einsum("ij,jk->ik", self.U, X_t) * self.σ + self.μ
        return np.array(X, dtype=np.float32)
'''


    





class ConstraintPreservingEmbedding():
    def __init__(self,A,b):
        self.A=A
        self.b=b
        self.orig_dim=self.A.shape[1]
        self.echelon,self.pivots=rref(np.concatenate((A,-b.reshape(b.shape[0],1)),axis=1))
        self.pivots=np.array(self.pivots)
        self.npivots=np.array(list(set(jnp.arange(self.orig_dim).tolist()).difference(self.pivots)))
    def transform(self,x):
        return jnp.delete(x, self.pivots, axis=0)

    def inverse_transform(self,x_red):
        tmp1=jnp.delete(self.echelon,self.pivots,axis=1)
        tmp2=jnp.concatenate((x_red,np.ones((1))))
        x_piv=-jnp.matmul(tmp1,tmp2)
        tmp3=jnp.zeros((self.orig_dim))
        tmp3[self.pivots]=x_piv
        tmp3[self.npivots]=x_red
        return tmp3
    
NUM_SAMPLES=600
tmp,barycenter=getinfo("rabbit_0.ply")
data=np.zeros((NUM_SAMPLES,*tmp.reshape(-1).shape))
for i in range(NUM_SAMPLES):
    data[i]=getinfo("rabbit_{}.ply".format(i))[0].reshape(-1)
data=jnp.array(data)
A=jnp.tile(jnp.eye(3),tmp.shape[0])/(tmp.shape[0])
vl=ConstraintPreservingEmbedding(A,barycenter)
tmparr=jnp.zeros((NUM_SAMPLES,vl.transform(data[0]).shape[0]))
for i in tqdm(range(NUM_SAMPLES)):
    A=jnp.tile(np.eye(3),tmp.shape[0])/(tmp.shape[0])
    vl=ConstraintPreservingEmbedding(A,barycenter)
    tmparr=tmparr.at[i].set(vl.transform(data[i]))

pca=PCA()
pca.fit(tmparr)
reduced=np.argmin(abs(np.cumsum(pca.explained_variance_ratio_)-(1-1e-5)))
print(reduced)
pca=PCA(reduced)
pca.fit(tmparr)
redarr=pca.transform(tmparr.copy())
print(jnp.linalg.norm(tmparr-pca.inverse_transform(redarr))/jnp.linalg.norm(tmparr))
pickle.dump(redarr,open("data.npy", 'wb'))


