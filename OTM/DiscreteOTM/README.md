# SISSA Internship working directory
This repo contains the file for implementing and Optimal transport methods.
Description of the two:
- compute_OTM is used to compute the tet6 of the optimal transport map between to meshes. Requires geogram.
- vorpalite contains some useful remeshing algorithms. Requires geogram.
- vorpastat calculate the Hausdorff distances between two meshes. Requires geogram
- *.stl are some solids I use for testing
- OTM.py calculates symbolically a continuos deformation map of the tet6 file (which is the output of compute_OTM and contanins the two meshes computes using the Optimal Transport Map) that preserves the volume in intermediate times, and writes the intermediate meshes to a series of vtk files
TO DO things:
- Test more complex geometries

