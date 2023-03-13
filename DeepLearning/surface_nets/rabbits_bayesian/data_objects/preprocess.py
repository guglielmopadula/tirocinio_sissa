 
import meshio
import numpy as np
import open3d as o3d
def reader(name):
    xyz = open(name)
    coords=[]    
    for line in xyz:
        x,y,z = line.split()
        coords.append([float(x),float(y),float(z)])
    xyz.close()
    coords=np.array(coords)
    return coords



points=reader("rabbit.xyz")
points[:,0]=points[:,0]-np.min(points[:,0])
points[:,1]=points[:,1]-np.min(points[:,1])
points[:,2]=points[:,2]-np.min(points[:,2])
points=points/np.max(points)



meshio.write_points_cells("rabbit.ply", points,[])


ply_point_cloud = o3d.data.PLYPointCloud()
pcd = o3d.io.read_point_cloud("rabbit.ply")
o3d.visualization.draw_geometries([pcd])    



