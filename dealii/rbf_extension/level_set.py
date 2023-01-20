"""Perform RBF extension of the semi-discrete optimal transport map to the whole domain."""

import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from scipy.stats.qmc import Halton
import numpy as np


cloud = np.load("sot_point_cloud.npy")
print("Cloud shape: ", cloud.shape)

state = cloud[:, :2]
additional_points = np.array([0, 0])
out_state = np.zeros(state.shape[0]+additional_points.shape[0])
out_state[-1] = additional_points

rbf = RBFInterpolator(np.vstack((state, additional_points)), out_state)

box_size = 1.5
xgrid = np.mgrid[-box_size:box_size:50j, -box_size:box_size:50j]
xflat = xgrid.reshape(2, -1).T

yflat = rbf(xflat)
fig, ax = plt.subplots(1, 2, figsize=(17, 7))

output = rbf(state) + state
print(output.shape)

ax.pcolormesh(*xgrid, yflat[:, i].reshape(50, 50), vmin=-0.25, vmax=0.25, shading='gouraud')
p = ax.scatter(state[:, 0], state[:, 1], c=displacements[:, i], s=50, ec='k', vmin=-0.25, vmax=0.25)
ax.scatter(output[:, 0], output[:, 1], c=displacements[:, i], s=50, ec='k', vmin=-0.25, vmax=0.25)
fig.colorbar(p, ax=ax)
ax.set_title("Transport map component {}".format(i))
plt.show()