import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from scipy.stats.qmc import Halton
import numpy as np


constraints = np.loadtxt("constraints.txt").reshape(-1, 3)
print("constraints shape: ", constraints.shape)

cloud = np.load("sot_point_cloud.npy")
print("Cloud shape: ", cloud.shape)

state_1 = cloud[:, :2]

box_size = 1.5
xgrid = np.mgrid[-box_size:box_size:50j, -box_size:box_size:50j]
xflat = xgrid.reshape(2, -1)

check = constraints[:, :2].dot(np.array([0.2, -0.1])) + constraints[:, 2]
mask_debug = check > 0
print("constraints pos", mask_debug.shape, constraints[mask_debug].shape)

yflat = constraints[:, :2].dot(xflat) + constraints[:, 2].reshape(-1, 1)
print(yflat.shape, yflat[mask_debug][0, :].shape)

boo_pos = yflat>0
boo_zero = yflat==0
mask = np.any(boo_pos, axis=0)
mask_zero = np.any(boo_zero, axis=0)
print(mask)
ynorm = np.linalg.norm(yflat, axis=0)
output = np.zeros(yflat.shape[1])
for i in range(yflat.shape[1]):
    if mask[i]:
        output[i] = 0
    else:
        output[i] = -1
    

fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax.pcolormesh(*xgrid, yflat[mask_debug][1, :].reshape(50, 50), vmin=-0.25, vmax=0.25, shading='gouraud')
p = ax.scatter(state_1[:, 0], state_1[:, 1], c=np.arange(state_1.shape[0]), s=50, ec='k', vmin=-0.25, vmax=0.25)
fig.colorbar(p, ax=ax)
ax.set_title("Transport map component")
plt.show()