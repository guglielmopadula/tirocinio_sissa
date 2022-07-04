#!/bin/bash
# Generate the series of numbers from 1 to 5
meshio convert cube.stl cube.ply
ctmconv cube.ply cube.ply
for i in {1..999}
do
	meshio convert cube_$i.stl cube_$i.ply
	ctmconv cube_$i.ply cube_$i.ply
	meshtorimd cube.ply cube_$i.ply cube_$i.txt
done
