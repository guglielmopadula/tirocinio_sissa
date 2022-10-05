"""Perform RBF extension of the semi-discrete optimal transport map to the whole domain."""

import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from scipy.stats.qmc import Halton
import numpy as np


cloud = np.load("sot_point_cloud.npy")
print("Cloud shape: ", cloud.shape)

state_1 = cloud[:, :2]
state_2 = cloud[:, 2:]

component = 1
displacements =  state_2 - state_1
rbf = RBFInterpolator(state_1, displacements)

box_size = 1.5
xgrid = np.mgrid[-box_size:box_size:50j, -box_size:box_size:50j]
xflat = xgrid.reshape(2, -1).T

yflat = rbf(xflat)
fig, ax = plt.subplots(1, 2, figsize=(17, 7))

output = rbf(state_1) + state_1
print(output.shape)

for i in range(2):
    ax[i].pcolormesh(*xgrid, yflat[:, i].reshape(50, 50), vmin=-0.25, vmax=0.25, shading='gouraud')
    p = ax[i].scatter(state_1[:, 0], state_1[:, 1], c=displacements[:, i], s=50, ec='k', vmin=-0.25, vmax=0.25)
    ax[i].scatter(output[:, 0], output[:, 1], c=displacements[:, i], s=50, ec='k', vmin=-0.25, vmax=0.25)
    fig.colorbar(p, ax=ax[i])
    ax[i].set_title("Transport map component {}".format(i))
plt.show()