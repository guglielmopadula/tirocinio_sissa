**Something to think of**

1. How to deal with singularities of the transport map ?
2. How to extend map from cloud of points to the whole domain ? RBF, Laplace extension
3. Test conservtation of volumes normalizing the transport map
4. Intrusive registration-based ROMs


**POD-NN-ROM using SOT between 2 states in 2d: sketch of the procedure**
1. compute OTM via geogram
2. convert to 2d morphing
3. convert to boundary morphing
4. export points to dealii
5. define extension map
6. define the shape levelset in dealii from a set of cloud points
7. sample snapshot from dealii with CutFEM
8. save snapshots in python
9. apply reverse extended transport map with RBF
10. apply SVD and check decay
    
11. *perform POD-NN

**TODOs**
1. get morphing of 2d boundaries in the case of more than 2 states eg. triangle, circle, star
