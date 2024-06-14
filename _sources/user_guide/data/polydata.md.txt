PolyData
========

Presentation
------------

A polydata object is a structure representing a 2D or 3D shape.

A polydata must be one of the three types:

- A point cloud (vertices)
- A wireframe mesh (vertices + edges)
- A triangle mesh (vertices + triangles)

:warning: for wireframe or triangle mesh, no isolated points are allowed. You can :

- ignore triangles and edges and consider shape as a point cloud, eventually encoding information about other structures as point_data or point_weigths
- remove unused points

Initialize a PolyData
---------------------

There are different ways to initialize a polydata:

- manually, providing vertices, edges, triangles as torch.tensors
```python
import skshapes as sks
import torch

# Manually set points
points = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 1],
    ],
    dtype=sks.float_dtype,
)

# Manually set points
triangles = torch.tensor(
    [
        [0, 1, 2],
        [0, 2, 3],
        [0, 1, 4],
        [1, 2, 4],
        [2, 3, 4],
        [3, 0, 4],
    ],
)

# Create shape
pyramid = sks.PolyData(points=points, triangles=triangles)
```
- from a file
```python
import skshapes as sks

# Load file
shape = sks.PolyData("mesh.vtk")
```
- from vedo.Mesh or pyvista.Polydata
```python
import pyvista.examples

# Load a pyvista PolyData from pyvista examples
bunny_pyvista = pyvista.examples.download_bunny()
# Cast it as a scikit-shapes PolyData
bunny_sks = sks.PolyData(bunny_pyvista)

import vedo
from vedo import dataurl

# Load a vedo Mesh from vedoexamples
pot = vedo.Mesh(dataurl + "teapot.vtk").shrink(0.75)
# Cast it as a scikit-shapes PolyData
pot_sks = sks.PolyData(pot)
```

Features
--------

- Some features can be computed : `edges_length`, `triangle_normals`, ...
- More complex features as `curvature`, `convolution`...

In addition to those features, you can add your own signals. The only restriction is that these signals must be defined point-wise
```python
import skshapes as sks
import torch

shape = sks.Circle
n_points = shape.n_points

shape["rnd_signal"] = torch.rand(n_points)
```
Note that the only restriction on the signal's shape is that the first dimension matches the number of points. There is no restriction about the number of dimensions and shapes as `(n_points, 2)`, `(n_points, 3, 3)` or `(n_points, 1, 2, 3, 4)` are valid.

Landmarks
---------

Landmarks are distinguished vertices. The main utility of defining landmarks is the ability to provide loss functions based on them.

Landmarks are represented as a sparse `torch.tensor`

Landmarks can be set following

Control points
--------------

Control points are

Multiscaling
------------

Multiscaling allows to represent a shape at different scales ensuring consistency of landmarks, control points and signal across scales. Read the documentation of Multiscaling to know more about this functionality.
