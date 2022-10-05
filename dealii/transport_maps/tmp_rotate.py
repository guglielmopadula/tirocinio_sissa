import numpy as np

points = np.array([
    [0, 1], 
    [0.86602540378, 0.5], 
    [0.86602540378, -0.5], 
    [0, -1], 
    [-0.86602540378, -0.5], 
    [-0.86602540378, 0.5], 
])

angle = np.pi / 6
rotation = np.array([
    [np.cos(angle), -np.sin(angle)],
    [np.sin(angle), np.cos(angle)]
])

print(0.5 * rotation.dot(points.T).T)