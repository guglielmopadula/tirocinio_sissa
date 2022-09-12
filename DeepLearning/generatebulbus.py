import numpy
from stl import mesh
import stl
your_mesh = mesh.Mesh.from_file('../Data/bulbo.stl')
volume, cog, inertia = your_mesh.get_mass_properties()
your_mesh.vectors=your_mesh.vectors-cog
your_mesh.vectors=your_mesh.vectors/volume**(1/3)
vec1=abs(numpy.random.normal(0,1,100))
vec2=abs(numpy.random.normal(0,1,100))
vec3=abs(numpy.random.normal(0,1,100))
for i in range(0,100):
    data = numpy.zeros(len(your_mesh), dtype=mesh.Mesh.dtype)
    data['vectors']=your_mesh.vectors
    data['vectors'][:,:,0]=data['vectors'][:,:,0]*(1+vec1[i])/(((1+vec1[i])*(1+vec2[i])*(1+vec3[i]))**(1/3))
    data['vectors'][:,:,1]=data['vectors'][:,:,1]*(1+vec2[i])/(((1+vec1[i])*(1+vec2[i])*(1+vec3[i]))**(1/3))
    data['vectors'][:,:,2]=data['vectors'][:,:,2]*(1+vec3[i])/(((1+vec1[i])*(1+vec2[i])*(1+vec3[i]))**(1/3))
    print((1+vec1[i])/(((1+vec1[i])*(1+vec2[i])*(1+vec3[i]))**(1/3)),(1+vec1[i])/(((1+vec2[i])*(1+vec2[i])*(1+vec3[i]))**(1/3)),(1+vec3[i])/(((1+vec3[i])*(1+vec2[i])*(1+vec3[i]))**(1/3)))
    mymesh = mesh.Mesh(data.copy())
    mymesh.save('bulbo_{}.stl'.format(i), mode=stl.Mode.ASCII)
