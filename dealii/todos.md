**Something to think of**

1. How to deal with singularities of the transport map ?
2. How to extend map from cloud of points to the whole domain ? RBF, Laplace extension
3. Test conservtation of volumes normalizing the transport map
4. Greedy definition of extended transport map to reduce evals decay
5. Use sensitivities of the PDE solution w.r.t geometric parameter to get better transport map extensions
6. Intrusive registration-based ROMs
7. Problem: introducing unsmooth features in reference domain affects eigenmodes quality?
   
**POD-NN-ROM using SOT between 2 states in 2d: sketch of the procedure**
1. compute OTM via geogram
2. convert to 2d morphing
3. extract boundary morphing
4. convert Delunay mesh to quads
5. define extension map from boundary morphing and ad hoc design for PDE domain
6. define the levelset in dealii from quad non-matching mesh
7. compute snapshot from dealii with CutFEM
8. create dealii deformation mapping from transport map
9. apply reverse extended transport map
10. apply SVD and check decay
    
11. *perform POD-NN

**TODOs Francesco**
1. Smooth out ghost domain with Laplace problem or filtering
2. Compare SVD decays w.r.t degree of FEM
3. Compare transported intermediate solutions with FOM and with ROM solutions

**TODOs theory**
1. Do geometry dependent analytical solutions of Poisson pb allign when pull-backed with transport map?

**TODOs Guglielmo-Francesco**
1. convert Delunay triangulation from geogram in quad grids for dealii
2. extract boundary morphing more accurately: problems with stretched boundary triangles
3. problem with too fine quad meshes from gmsh
4. get morphing in the case of more than 2 states eg. triangle, circle, star
