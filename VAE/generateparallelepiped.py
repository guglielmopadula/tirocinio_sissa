from stl import mesh
import math
import numpy
import stl


def createcube(a,b,name):
    data = numpy.zeros(12, dtype=mesh.Mesh.dtype)
    c=1/(a*b)

# x=0
    data['vectors'][0] = numpy.array([[0, 0, 0],
                                  [0, b, 0],
                                  [0, 0, c]])
    data['vectors'][1] = numpy.array([[0, b, c],
                                  [0, b, 0],
                                  [0, 0, c]])


# x=1 
    data['vectors'][2] = numpy.array([[a, 0, 0],
                                  [a, b, 0],
                                  [a, 0, c]])
    data['vectors'][3] = numpy.array([[a, b, c],
                                  [a, b, 0],
                                  [a, 0, c]])


#y=0
    data['vectors'][4] = numpy.array([[0, 0, 0],
                                  [a, 0, 0],
                                  [0, 0, c]])
    data['vectors'][5] = numpy.array([[a, 0, c],
                                  [a, 0, 0],
                                  [0, 0, c]])


#y=1
    data['vectors'][6] = numpy.array([[0, b, 0],
                                  [a, b, 0],
                                  [0, b, c]])
    data['vectors'][7] = numpy.array([[a, b, c],
                                  [a, b, 0],
                                  [0, b, c]])

 #z=0
    data['vectors'][8] = numpy.array([[0, 0, 0],
                                  [a, 0, 0],
                                  [0, b, 0]])
    data['vectors'][9] = numpy.array([[a, b, 0],
                                  [a, 0, 0],
                                  [0, b, 0]])

 #z=1
    data['vectors'][10] = numpy.array([[0, 0, c],
                                  [a, 0, c],
                                  [0, b, c]])
    data['vectors'][11] = numpy.array([[a, b, c],
                                  [a, 0, c],
                                  [0, b, c]])




    data['vectors']=data['vectors']-numpy.array([a/2,b/2,c/2])
    mymesh = mesh.Mesh(data.copy())
    mymesh.save(name+'.stl', mode=stl.Mode.ASCII)


vec1=abs(numpy.random.normal(3,0.5,1000))
vec2=abs(numpy.random.normal(3,0.5,1000))
createcube(vec1[0],vec2[0],"cube")
for i in range(1,1000):
	createcube(vec1[i],vec2[i],"parallelepiped_{}".format(i))
