//+
Point(1) = {0, 1, 0, 1.0};
//+
Point(2) = {0.86602540378, -0.5, 0, 1.0};
//+
Point(3) = {-0.86602540378, -0.5, 0, 1.0};
//+
Line(1) = {1, 3};
//+
Line(2) = {3, 2};
//+
Line(3) = {2, 1};
//+
Curve Loop(1) = {1, 2, 3};
//+
Surface(1) = {1};
//+
Extrude {0, 0, 1} {
  Surface{1}; 
}
//+
Physical Surface(1) = {20};
