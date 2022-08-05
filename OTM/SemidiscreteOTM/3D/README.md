# SISSA Internship working directory
This repo contains the file for implementing and Optimal transport methods.
Description of the two:
- OTM3D.py calculates numerically (howewer exactly) a continuos deformation map of the tet6 file (which is the output of compute_OTM and contains the two 3D meshes computed using the Optimal Transport Map) that preserves the volume in intermediate times, and writes the intermediate meshes to a series of vtk files.
- double_OTM3D.py calculates numerically (howewer exactly) a continuos deformation map of two tet6 file (which are the output of double_OTM and contanins the three 3D meshes computes using the Optimal Transport Map) that preserves the volume in intermediate times, and writes the intermediate meshes to a series of vtk files.

