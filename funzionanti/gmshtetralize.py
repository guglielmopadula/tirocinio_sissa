import gmsh
gmsh.initialize()

stl = gmsh.merge('cube.stl') # 3D STL file of a cylinder
gmsh.model.mesh.classifySurfaces(gmsh.pi, True, True, gmsh.pi)
gmsh.model.mesh.createGeometry()
s = gmsh.model.getEntities(2)
surf = gmsh.model.geo.addSurfaceLoop([e[1] for e in s])
vol = gmsh.model.geo.addVolume([surf])
gmsh.model.geo.synchronize()

gmsh.option.setNumber('Mesh.Algorithm', 1)
gmsh.option.setNumber('Mesh.MeshSizeMax', 0.01)
gmsh.option.setNumber('Mesh.MeshSizeMin', 0.01)
gmsh.model.mesh.generate(3)
gmsh.write('cube.mesh')
