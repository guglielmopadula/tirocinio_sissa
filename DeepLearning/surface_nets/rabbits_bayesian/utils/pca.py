import jax.numpy as jnp
from jax import jit
import jax
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)


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
    #return jnp.einsum("ik,kj->ij", data, U, precision=jax.lax.Precision.HIGH)
    return data@U
@jit
def _inverse_transform(data,mean,U):
    #print(U.T.shape)
    tmp=data@U.T
    return tmp+mean
    #return jnp.einsum("ik,kj->ij", data, U.T, precision=jax.lax.Precision.HIGH)+mean






