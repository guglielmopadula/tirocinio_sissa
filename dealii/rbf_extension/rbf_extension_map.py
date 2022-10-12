"""Perform RBF extension of the semi-discrete optimal transport map to the whole domain."""

import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from scipy.stats.qmc import Halton
import numpy as np


class RBF(object):
    def __init__(self, controlp, coeffs, shift, scale, powers):
        self.coeffs = coeffs
        self.shift = shift
        self.scale = scale
        self.controlp = controlp
        self.powers = powers
        self.rbf_basis = np.vectorize(self.thin_plate_spline)

        print(controlp.shape,
              coeffs.shape,
              shift.shape,
              scale.shape,
              powers.shape)
        np.savetxt("../step-85/controlpx.txt", self.controlp[:, 0])
        np.savetxt("../step-85/controlpy.txt", self.controlp[:, 1])
        np.savetxt("../step-85/coeffs.txt", self.coeffs.reshape(-1))
        np.savetxt("../step-85/shift.txt", self.shift.reshape(-1))
        np.savetxt("../step-85/scale.txt", self.scale.reshape(-1))
        np.savetxt("../step-85/powersx.txt", self.powers[:, 0])
        np.savetxt("../step-85/powersy.txt", self.powers[:, 1])

    def __call__(self, x):
        assert(x.shape[0] == 1)
        d = self.controlp.shape[0]
        p = self.coeffs.shape[0]
        vec = np.ones(p)
        xhat = (x - self.shift)/self.scale

        vec[:d] = self.rbf_basis(np.linalg.norm(
            x.reshape(1, -1) - self.controlp, axis=1))

        vec[d:] = np.prod(xhat.reshape(1, -1)**self.powers, axis=1)
        return vec.dot(self.coeffs)

    @staticmethod
    def thin_plate_spline(r):
        if r == 0:
            return 0.0
        else:
            return r**2*np.log(r)


cloud = np.load("sot_point_cloud.npy")
print("Cloud shape: ", cloud.shape)

state_1 = cloud[:, :2]
state_2 = cloud[:, 2:]

component = 1
displacements = state_2 - state_1

box_size = 1.5
xgrid = np.mgrid[-box_size:box_size:50j, -box_size:box_size:50j]
xflat = xgrid.reshape(2, -1).T

# additional points on boundary
eps = 0.001
n_boundary = 100
bset = np.linspace(-1+eps, 1-eps, n_boundary).reshape(-1, 1)

ones = np.ones((bset.shape[0], 1))
boundary_points = np.vstack([
    np.hstack((1.5*bset, 1.5*ones)),
    np.hstack((1.5*bset, -1.5*ones)),
    np.hstack((1.5*ones, 1.5*bset)),
    np.hstack((-1.5*ones, 1.5*bset))
])

# additional points close to boundary
eps_b = 0.06
n_close = 50
bset = np.linspace(-1+eps, 1-eps, n_close).reshape(-1, 1)

ones = np.ones((bset.shape[0], 1))-eps_b
close_points = np.vstack([
    np.hstack((1.5*bset, 1.5*ones)),
    np.hstack((1.5*bset, -1.5*ones)),
    np.hstack((1.5*ones, 1.5*bset)),
    np.hstack((-1.5*ones, 1.5*bset))
])

boundary_displacements = np.zeros((boundary_points.shape[0], 2))
close_displacements = np.zeros((close_points.shape[0], 2))
state_1 = np.vstack((state_1, boundary_points, close_points))
displacements = np.vstack((displacements, boundary_displacements, close_displacements))

rbf = RBFInterpolator(state_1, displacements)
rbf_test = RBF(state_1, rbf._coeffs, rbf._shift, rbf._scale, rbf.powers)

yflat = rbf(xflat)
fig, ax = plt.subplots(1, 2, figsize=(17, 7))

output = rbf(state_1) + state_1
output_test = rbf_test(state_1[0].reshape(1, -1)) + state_1[0].reshape(1, -1)
print(output.shape, output_test.shape, output[:3], output_test[:3])

for i in range(2):
    ax[i].pcolormesh(*xgrid, yflat[:, i].reshape(50, 50),
                     vmin=-0.25, vmax=0.25, shading='gouraud')
    p = ax[i].scatter(state_1[:, 0], state_1[:, 1],
                      c=displacements[:, i], s=50, ec='k', vmin=-0.25, vmax=0.25)
    ax[i].scatter(output[:, 0], output[:, 1], c=displacements[:,
                  i], s=50, ec='k', vmin=-0.25, vmax=0.25)
    fig.colorbar(p, ax=ax[i])
    ax[i].quiver(*xgrid, yflat[:, 0].reshape(50, 50),
                 yflat[:, 1].reshape(50, 50))
    ax[i].set_title("Transport map component {}".format(i))
plt.show()
