from smithers.io import STLHandler
from smithers.io import VTPHandler
from pygem import FFD
import numpy as np

fname = 'CSG.stl'

ffd = FFD([3, 3, 4])
ffd.box_origin = np.array([-2, 0., -0.4])
ffd.box_length = np.array([-0.7, 0.15, 0.5])
data = STLHandler.read(fname)
points = {'points': ffd.control_points(False), 'cells': [], 'point_data': {},'cell_data': {}}
VTPHandler.write('points.vtp', points)
data['points'] = ffd(data['points'])
STLHandler.write('CSG_transformed.stl', data)
