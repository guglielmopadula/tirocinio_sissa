import argparse
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from jax import vmap
import jax.numpy as jnp
import jax.random as random
import meshio
import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from datawrapper.data import Data
from models.BAE import BAE
NUM_TRAIN=100
NUM_TEST=100
NUM_SAMPLES=NUM_TRAIN+NUM_TEST
data=Data(num_test=NUM_TEST,num_train=NUM_TRAIN,string="data_objects/rabbit_{}.ply",red_dim=140, batch_size=NUM_TRAIN)

moment_tensor_data=np.zeros((NUM_SAMPLES,3,3))
for j in range(3):
    for k in range(3):
        moment_tensor_data[:,j,k]=np.mean(data.data.reshape(NUM_SAMPLES,-1,3)[:,:,j]*data.data.reshape(NUM_SAMPLES,-1,3)[:,:,k],axis=1)


model=BAE(pca_mean=data.pca.mean,barycenter=data.barycenter,pca_V=data.pca.U,latent_dim=20,hidden_dim=30)



def run_inference(model, rng_key, X):
    start = time.time()
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=500,
        num_samples=NUM_SAMPLES,
        num_chains=1,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(rng_key, X)
    mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()


def predict(model, rng_key, samples, X):
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    model_trace = handlers.trace(model).get_trace(X=X)
    return model_trace["obs"]["value"]

rng_key, rng_key_predict = random.split(random.PRNGKey(0))
new_samples = run_inference(model, rng_key, data.data_train)[:,0,:]
print(new_samples.shape)
moment_tensor_sampled=np.zeros((NUM_SAMPLES,3,3))

name="BAE"
for i in range(NUM_SAMPLES):
    tmp=new_samples[i].reshape(-1,3)
    meshio.write_points_cells("./inference_objects/"+name+"_{}.ply".format(i), tmp,[])    
    for j in range(3):
        for k in range(3):
            moment_tensor_sampled[:,j,k]=np.mean(tmp.reshape(-1,3)[:,:,j]*tmp.reshape(NUM_SAMPLES,-1,3)[:,:,k],axis=1)


fig2,ax2=plt.subplots()
ax2.set_title("XX moment of "+name)
_=ax2.hist([moment_tensor_data[:,0,0].reshape(-1),moment_tensor_sampled[:,0,0].reshape(-1)],8,label=['real','sampled'])
ax2.legend()
fig2.savefig("./inference_graphs/XXaxis_hist_"+name+".png")
fig2,ax2=plt.subplots()
ax2.set_title("YY moment of "+name)
_=ax2.hist([moment_tensor_data[:,1,1].reshape(-1),moment_tensor_sampled[:,1,1].reshape(-1)],8,label=['real','sampled'])
ax2.legend()
fig2.savefig("./inference_graphs/YYaxis_hist_"+name+".png")
fig2,ax2=plt.subplots()
ax2.set_title("ZZ moment of "+name)
_=ax2.hist([moment_tensor_data[:,2,2].reshape(-1),moment_tensor_sampled[:,2,2].reshape(-1)],8,label=['real','sampled'])
ax2.legend()
fig2.savefig("./inference_graphs/ZZaxis_hist_"+name+".png")
fig2,ax2=plt.subplots()
ax2.set_title("XY moment of "+name)
_=ax2.hist([moment_tensor_data[:,0,1].reshape(-1),moment_tensor_sampled[:,0,1].reshape(-1)],8,label=['real','sampled'])
ax2.legend()
fig2.savefig("./inference_graphs/XYaxis_hist_"+name+".png")
fig2,ax2=plt.subplots()
ax2.set_title("XZ moment of "+name)
_=ax2.hist([moment_tensor_data[:,0,2].reshape(-1),moment_tensor_sampled[:,0,2].reshape(-1)],8,label=['real','sampled'])
ax2.legend()
fig2.savefig("./inference_graphs/XZaxis_hist_"+name+".png")
fig2,ax2=plt.subplots()
ax2.set_title("YZ moment of "+name)
_=ax2.hist([moment_tensor_data[:,1,2].reshape(-1),moment_tensor_sampled[:,1,2].reshape(-1)],8,label=['real','sampled'])
ax2.legend()
fig2.savefig("./inference_graphs/YZaxis_hist_"+name+".png")
