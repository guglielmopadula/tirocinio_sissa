import numpy
from stl import mesh
import stl
your_mesh = mesh.Mesh.from_file('bulbo.stl')
volume, cog, inertia = your_mesh.get_mass_properties()
your_mesh.vectors=your_mesh.vectors-cog
vec1=abs(numpy.random.normal(0,1,100))
vec2=abs(numpy.random.normal(0,1,100))
vec3=abs(numpy.random.normal(0,1,100))

data = numpy.zeros(len(your_mesh), dtype=mesh.Mesh.dtype)
data['vectors']=your_mesh.vectors
data['vectors'][:,:,0]=data['vectors'][:,:,0]*(4/3)
data['vectors'][:,:,1]=data['vectors'][:,:,1]*(3/4)
data['vectors'][:,:,2]=data['vectors'][:,:,2]
mymesh = mesh.Mesh(data.copy())
mymesh.save('bulbo_a.stl',mode=stl.Mode.ASCII)


data = numpy.zeros(len(your_mesh), dtype=mesh.Mesh.dtype)
data['vectors']=your_mesh.vectors
data['vectors'][:,:,0]=data['vectors'][:,:,0]*(3/4)
data['vectors'][:,:,1]=data['vectors'][:,:,1]*(4/3)
data['vectors'][:,:,2]=data['vectors'][:,:,2]
mymesh = mesh.Mesh(data.copy())
mymesh.save('bulbo_b.stl',mode=stl.Mode.ASCII)


data = numpy.zeros(len(your_mesh), dtype=mesh.Mesh.dtype)
data['vectors']=your_mesh.vectors
data['vectors'][:,:,0]=data['vectors'][:,:,0]*(4/3)
data['vectors'][:,:,1]=data['vectors'][:,:,1]
data['vectors'][:,:,2]=data['vectors'][:,:,2]*(3/4)
mymesh = mesh.Mesh(data.copy())
mymesh.save('bulbo_c.stl',mode=stl.Mode.ASCII)
