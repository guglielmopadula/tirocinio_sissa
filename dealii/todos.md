**Something to think of**

1. How to deal with singularities of the transport map ?
2. How to extend map from cloud of points to the whole domain ? RBF, Laplace extension
3. Test conservtation of volumes normalizing the transport map
4. Greedy definition of extended transport map to reduce evals decay
5. Use sensitivities of the PDE solution w.r.t geometric parameter to get better transport map extensions
6. Intrusive registration-based ROMs


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

**TODOs**
1. convert Delunay triangulation from geogram in quad grids for dealii
2. extract boundary morphing more accurately

Less urgent
3. get morphing of 2d boundaries in the case of more than 2 states eg. triangle, circle, star
4. shorten the number of boundary vertices that define with a sufficiently good accuracy all the intermediate states of sot
