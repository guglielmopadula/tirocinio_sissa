import numpy as np
import matplotlib.pyplot as plt

# mesh parsing
from smithers.io.openfoam import OpenFoamHandler
from smithers.io import STLHandler
# interpolator
from scipy.interpolate import Rbf
import sys
from pygem import FFD, RBF

def scatter3d(arr, figsize=(8,8), s=10, draw=True, ax=None, alpha=1, labels=None, hull=None):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')
        
        ax.scatter(*arr[0].T, s=s, alpha=alpha, label=labels[0])
        ax.scatter(*arr[1].T, s=10, alpha=alpha, label=labels[1])
    
    if hull is not None:
        ax.scatter(*hull.T, s=s, alpha=alpha)
        
    if draw:
        if labels is not None:
            plt.legend()
        plt.show()
    else:
        return ax

def expand_to_surface(f1, f2, p3, dim):
        l = [0, 1, 2]
        l.remove(dim)
        xx, yy = np.meshgrid(f1, f2)
        t = np.zeros((xx.reshape(-1).shape[0], 3))
        t[:, l[0]] = xx.reshape(-1)
        t[:, l[1]] = yy.reshape(-1)
        t[:, dim] = p3
        return t

def filter(dat, lim):
    filtered = np.logical_and.reduce((
        dat[:, 0] < lim[1][0],
        dat[:, 0] > lim[0][0],
        dat[:, 1] < lim[1][1],
        dat[:, 1] > lim[0][1],
        dat[:, 2] < lim[1][2],
        dat[:, 2] > lim[0][2])
    )
    return filtered

def filter_and_add_boundaries(dataset, fix, limits):
    filtered = filter(dataset, limits)
    points = dataset[filtered]
    points = np.vstack((points, fix))
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    # plt.show()
    
    return points

box = np.array([[5.2, -0.2, -0.1],
                [6.25, 0.02, 0.3]])

n = 16
x = np.linspace(0, 1, n)
x = np.tile(x, (3)).reshape(3, -1).T
f = box[0].reshape(1, -1)+x*(box[1]-box[0]).reshape(1, -1)

ff = []
ff.append(expand_to_surface(f[:, 0], f[:, 1], box[0, 2], 2))
ff.append(expand_to_surface(f[:, 0], f[:, 1], box[1, 2], 2))
ff.append(expand_to_surface(f[:, 1], f[:, 2], box[0, 0], 0))
ff.append(expand_to_surface(f[:, 1], f[:, 2], box[1, 0], 0))
ff.append(expand_to_surface(f[:, 0], f[:, 2], box[0, 1], 1))
# ff.append(expand_to_surface(f[:, 0], f[:, 2], box[1, 1], 1))

fixed = np.vstack(ff)

openfoam_handler = OpenFoamHandler()

# with open("./sample/constant/polyMesh/points", 'r') as file:
#     with open("./points_only", 'w') as outfile:
#         data = file.read()
#         data = data.replace("(", " ")
#         data = data.replace(")", " ")
#         outfile.write(data)
                
mesh = np.loadtxt("./points_only", skiprows=21, max_rows=918334)
print("mesh: ", mesh.shape, mesh[0], mesh[-1])

mesh_filter = filter(mesh, box)
mesh_filtered = mesh[mesh_filter]#[::100]
print("filtered mesh: ", mesh_filtered.shape)

fname = "./deformations/hull/hull_rotated_0.stl"
reference = STLHandler.read(fname)['points']
print(reference.shape,
      np.max(reference[:, 0]),
      np.min(reference[:, 0]),
      np.max(reference[:, 1]),
      np.min(reference[:, 1]),
      np.max(reference[:, 2]),
      np.min(reference[:, 2]))

undeformed_points = filter_and_add_boundaries(reference, fixed, box)
undeformed_points, uindexes = np.unique(undeformed_points, return_index=True, axis=0)

for i in range(100):
    print(i, sys.argv[1])
    fname = "./deformations/"+sys.argv[1]+"/"+sys.argv[1]+"_rotated_"+str(i)+".stl"
    data = STLHandler.read(fname)['points']
    # print(data.shape)

    deformed_points   = filter_and_add_boundaries(data, fixed, box)
    deformed_points = deformed_points[uindexes]
        
    rbf = RBF(original_control_points=undeformed_points, deformed_control_points=deformed_points, func='thin_plate_spline', radius=1)

    # scatter3d([rbf.original_control_points, rbf.deformed_control_points], s=10, labels=['original', 'deformed'])

    new_mesh_points = rbf(rbf.original_control_points)
    err = np.max(np.abs(new_mesh_points-rbf.deformed_control_points))/np.max(np.abs(rbf.original_control_points))
    print("Relative error: ", err)

    new_mesh_points = rbf(mesh_filtered)
    eps = 1e-3
    ebox = np.array([[box[0, 0]-eps, box[0, 1]-eps, box[0, 2]-eps],
                     [box[1, 0]+eps, box[1, 1]+eps, box[1, 2]+eps]])
    epoints = new_mesh_points[~filter(new_mesh_points, ebox)]
    fl_out = epoints.shape[0]>0
    # scatter3d([mesh_filtered, epoints], s=10, labels=['original', 'deformed'])
    print("Check all points inside box: ", fl_out, epoints.shape[0])
    
    if err > 1e-4 or fl_out :
        print("CRITERIA NOT MATCHED")
        break
    
    # cmesh = np.copy(mesh)
    # cmesh[mesh_filter] = new_mesh_points
    idx_filtered = np.arange(mesh.shape[0])[mesh_filter]+21
    print(mesh.shape, idx_filtered.shape, new_mesh_points.shape, mesh_filtered.shape)
    # np.savetxt("./points.txt", cmesh)
    idx = 0
    with open("./sample/constant/polyMesh/points", 'r') as file:
        with open("./"+sys.argv[1]+"/points_"+str(i), 'w') as outfile:
            data = file.readlines()
            for i_line in range(idx_filtered.shape[0]):
                data[idx_filtered[i_line]] = "({} {} {})\n".format(new_mesh_points[idx, 0],new_mesh_points[idx, 1], new_mesh_points[idx, 2])
            outfile.writelines(data)
    print("end ", i)
    
    # scatter3d([mesh, epoints], s=0.5, labels=['original', 'deformed'], hull=rbf.original_control_points)
    plt.show()

# sys.exit()
