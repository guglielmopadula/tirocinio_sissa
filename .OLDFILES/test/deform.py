from smithers.io import STLHandler
from smithers.io import VTPHandler
from pygem import FFD
import numpy as np

fname = 'Parallelepiped.stl'

ffd = FFD([3, 3, 4])
ffd.box_origin = np.array([-2, 0., -0.4])
ffd.box_length = np.array([-0.7, 0.15, 0.5])

for i in range(1):
    param = np.random.uniform(0, 0.5, 6)-0.25
    print(param)
    ffd.reset_weights()
    
    data = STLHandler.read(fname)

    points = {'points': ffd.control_points(False), 'cells': [], 'point_data': {},
            'cell_data': {}}
    VTPHandler.write('points.vtp', points)

    data['points'] = ffd(data['points'])
    STLHandler.write('front{}.stl'.format(i), data)
