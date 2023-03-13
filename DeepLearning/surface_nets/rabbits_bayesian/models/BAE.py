import numpyro
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import jit 

def nonlin(x):
    return jnp.tanh(x)


# a two-layer bayesian neural network with computational flow
# given by D_X => D_H => D_H => D_Y where D_H is the number of
# hidden units. (note we indicate tensor dimensions in the comments)
def BAE(pca_mean,pca_V,barycenter,latent_dim,hidden_dim):
    def model(x):
        N = x.shape[0]
        D= x.shape[1]
        x_dim=pca_V.shape[1]
        x_red=jnp.matmul(x-pca_mean,pca_V)        
        w1 = numpyro.sample("w1", dist.Normal(jnp.zeros((x_dim, hidden_dim)), jnp.ones((x_dim, hidden_dim))))
        z1 = nonlin(jnp.matmul(x_red, w1))  

        w2 = numpyro.sample("w2", dist.Normal(jnp.zeros((hidden_dim, hidden_dim)), jnp.ones((hidden_dim, hidden_dim))))
        z2 = nonlin(jnp.matmul(z1, w2))  

        w3 = numpyro.sample("w3", dist.Normal(jnp.zeros((hidden_dim, latent_dim)), jnp.ones((hidden_dim, latent_dim))))
        z3 = nonlin(jnp.matmul(z2, w3)) 

        w4 = numpyro.sample("w4", dist.Normal(jnp.zeros((latent_dim, hidden_dim)), jnp.ones((latent_dim, hidden_dim))))
        z4 = nonlin(jnp.matmul(z3, w4))  

        w5 = numpyro.sample("w5", dist.Normal(jnp.zeros((hidden_dim, hidden_dim)), jnp.ones((hidden_dim, hidden_dim))))
        z5 = nonlin(jnp.matmul(z4, w5))  

        w6 = numpyro.sample("w6", dist.Normal(jnp.zeros((hidden_dim, x_dim)), jnp.ones((hidden_dim, x_dim))))
        z6 = jnp.matmul(z5, w6)  

        x_hat=z6@pca_V.T+pca_mean
        x_hat=x_hat.reshape(N,-1,3)
        x_hat=(x_hat-jnp.mean(x_hat,axis=1).expand_dims(1).repeat(1,D//3,1)+(barycenter.expand_dims(0)).expand_dims(0).repeat(N,D//3,1))
        x_hat=x_hat.reshape(N,D)
        # observe data
        with numpyro.plate("data", N):
            # note we use to_event(1) because each observation has shape (1,)
            numpyro.sample("obs", dist.Normal(z3, x_hat).to_event(1), obs=x)
    return model