import argparse
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from jax import vmap
import jax.numpy as jnp
import jax.random as random

import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from datawrapper.data import Data
from models.BAE import BAE
NUM_TRAIN=300
NUM_TEST=300

data=Data(num_test=NUM_TEST,num_train=NUM_TRAIN,string="data_objects/rabbit_{}.ply",red_dim=140, batch_size=NUM_TRAIN)
model=BAE(pca_mean=data.pca.mean,barycenter=data.barycenter,pca_V=data.pca.U,latent_dim=20,hidden_dim=30)


def run_inference(model, rng_key, X):
    start = time.time()
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=500,
        num_samples=1,
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
samples = run_inference(model, rng_key, data.data_train)