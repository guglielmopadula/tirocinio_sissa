from smithers.io import STLHandler
from smithers.io import VTPHandler
from pygem import FFD
import numpy as np

fname = 'DTMB_per_giovanni_front.stl'



ffd = FFD([3, 3, 4])
ffd.box_origin = np.array([-2, 0., -0.4])
ffd.box_length = np.array([-0.7, 0.15, 0.5])

for i in range(40):
    param = np.random.uniform(0, 0.5, 6)-0.25
    print(param)
    ffd.reset_weights()

    ffd.array_mu_x[2, 2, 1] += param[0]*2
    ffd.array_mu_x[2, 2, 0] += param[0]*2
    ffd.array_mu_x[2, 1, 1] += param[0]*2
    ffd.array_mu_x[2, 1, 0] += param[0]*2
    ffd.array_mu_x[2, 0, 1] += param[0]*2
    ffd.array_mu_x[2, 0, 0] += param[0]*2

    ffd.array_mu_y[2, 2, 1] += param[1]
    ffd.array_mu_y[2, 2, 0] += param[1]

    ffd.array_mu_z[2, 2, 1] += param[5]
    ffd.array_mu_z[2, 2, 0] += param[5]

    ffd.array_mu_z[1, 2, 1] += param[2]
    ffd.array_mu_y[1, 2, 1] += param[3]

    ffd.array_mu_z[2, 1, 1] += param[4]
    ffd.array_mu_z[2, 1, 2] += param[4]

    data = STLHandler.read(fname)

    points = {'points': ffd.control_points(False), 'cells': [], 'point_data': {},
            'cell_data': {}}
    VTPHandler.write('points.vtp', points)

    data['points'] = ffd(data['points'])
    STLHandler.write('front{}.stl'.format(i), data)
