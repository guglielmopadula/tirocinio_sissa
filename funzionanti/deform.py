from smithers.io import STLHandler
from smithers.io import VTPHandler
from pygem import FFD
import numpy as np

fname = 'cube.stl'



ffd = FFD([2, 2, 2])
ffd.box_origin = np.array([0, 0, 0])
ffd.box_length = np.array([1, 1, 1])



ffd.array_mu_x[0, 1, 0] = 2
ffd.array_mu_x[1, 0, 0] = 2
data = STLHandler.read(fname)
points = {'points': ffd.control_points(False), 'cells': [], 'point_data': {},
            'cell_data': {}}
VTPHandler.write('points.vtp', points)
data['points'] = ffd(data['points'])
STLHandler.write('test.stl', data)
