#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 12:46:22 2023

@author: cyberguli
"""

import trimesh
mesh00=trimesh.load_mesh("data_objects/hullmoved.stl")
mesh1p=trimesh.intersections.slice_mesh_plane(mesh00,(1,0,0),(0,0,0))
mesh1p.export("test_1p.stl")
mesh1n=trimesh.intersections.slice_mesh_plane(mesh00,(-1,0,0),(0,0,0))
mesh1n.export("test_1n.stl")
mesh1p1p=trimesh.intersections.slice_mesh_plane(mesh1p,(0,0,1),(0,0,0))
mesh1p1p.export("test_1p1p.stl")
mesh1p1n=trimesh.intersections.slice_mesh_plane(mesh1p,(0,0,-1),(0,0,0))
mesh1p1n.export("test_1p1n.stl")
mesh1p1p=trimesh.intersections.slice_mesh_plane(mesh1n,(0,0,1),(0,0,0))
mesh1p1p.export("test_1n1p.stl")
mesh1p1n=trimesh.intersections.slice_mesh_plane(mesh1n,(0,0,-1),(0,0,0))
mesh1p1n.export("test_1n1n.stl")
_,args=trimesh.boolean.union([mesh1p,mesh1n],engine="blender")