from smithers.io import STLHandler
from pygem import FFD
import numpy as np
np.random.seed(0)



def volume_tetra(M):
    return abs(np.linalg.det(M))/6


def volume(mesh):
    volume=0
    for i in range(len(mesh)):
        volume=volume+volume_tetra(mesh[i,:,:])
    return volume

fname='../../Data/bulbo_norm.stl'

ffd = FFD([6, 6, 6])
ffd.box_origin = np.array([-1.5, -1, -0.5])
ffd.box_length = np.array([3, 2, 1])
tempx=ffd.array_mu_x
tempy=ffd.array_mu_y
tempz=ffd.array_mu_z
for i in range(500):
    ffd.reset_weights()
    for a in range(6):
        for b in range(6):
            for c in range(6):
                param = np.random.uniform(-0.5, 0.5, 3)
                ffd.array_mu_x[a,b,c] = tempx[a,b,c]+param[0]
                ffd.array_mu_y[a,b,c] = tempy[a,b,c]+param[1]
                ffd.array_mu_z[a,b,c] = tempz[a,b,c]+param[2]
    data = STLHandler.read(fname)
    cells=data['cells']
    data['points'] = ffd(data['points'])
    M=data['points'][cells]
    data['points']=data['points']/volume(M)**(1/3)
    data['points'][:,0]=data['points'][:,0]-np.mean(data['points'][:,0])
    data['points'][:,1]=data['points'][:,1]-np.mean(data['points'][:,1])
    data['points'][:,2]=data['points'][:,2]-np.mean(data['points'][:,2])

    STLHandler.write('bulbo_{}.stl'.format(i), data)


