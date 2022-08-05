from smithers.io import STLHandler
from smithers.io import VTPHandler
from pygem import FFD
import numpy as np

fname = 'bulbo_orig.stl'



ffd = FFD([5, 5, 5])
ffd.box_origin = np.array([-60, 4, 2])
ffd.box_length = np.array([8, 2, 2])

for i in range(40):
    param = np.random.uniform(0, 0.5, 6)-0.25
    print(param)
    ffd.reset_weights()

    ffd.array_mu_y+= 5*param[0]


    data = STLHandler.read(fname)

    points = {'points': ffd.control_points(False), 'cells': [], 'point_data': {},
            'cell_data': {}}
    VTPHandler.write('points.vtp', points)

    data['points'] = ffd(data['points'])
    STLHandler.write('front{}.stl'.format(i), data)
