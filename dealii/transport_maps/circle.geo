Merge "circle.gmsh";
SetFactory("OpenCASCADE");
Circle(1) = {0, 0, 0, 1, 0, 2*Pi};
//+
Curve Loop(1) = {1};
//+
Surface(1) = {1};
//+
Extrude {0, 0, 1} {
  Surface{1}; 
}
//+
Physical Surface(1) = {3};
