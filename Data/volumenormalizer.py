#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 21:27:23 2022

@author: cyberguli
"""

from stl import mesh
import stl
import numpy as np
file="broken_sphere.stl"
your_mesh = mesh.Mesh.from_file(file)
M=your_mesh.vectors


def volume_tetra(M):
    return abs(np.linalg.det(M))/6


def volume(mesh):
    volume=0
    for i in range(len(mesh)):
        volume=volume+volume_tetra(mesh[i,:,:])
    return volume



print(volume(M))
M=M/volume(M)**(1/3)
data = np.zeros(len(M), dtype=mesh.Mesh.dtype)
data['vectors'] = M
mymesh = mesh.Mesh(data.copy())
mymesh.save(file, mode=stl.Mode.ASCII)
