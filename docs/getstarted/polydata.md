## Presentation

A polydata object is a structure representing a 2D or 3D shape.

A polydata must be one of the three types :
- A point cloud (vertices)
- A wireframe mesh (vertices + edges)
- A triangle mesh (vertices + triangles)

/!\ for wireframe or triangle mesh, no isolated points are allowed. You can :
- ignore triangles and edges and consider shape as a point cloud, eventually encoding information about other structures as point_data or point_weigths
- remove unused points

## Initialize a polydata :

 There are different ways to initialize a polydata:

- manually, providing vertices, edges, triangles as torch.tensors :
- from a file
- from vedo.Mesh or pyvista.Polydata (??)

## Landmarks

Landmarks are distinguished vertices. The main utility of defining landmarks is the ability to provide loss functions based on them.

Landmarks are represented as a sparse `torch.tensor`

Landmarks can be set following 

## Signals, or point data

Signals refered to quantities that can be defined

- triangle-wise : normals, centers, areas
- edge-wise : edge centers, lenghts
- point-wise : 

The modules features and curvatures gather other signals that are available in scikit-shapes.

You can also define your own pointwise signals 
>>> mesh["my_signal"] = torch.rand(mesh.n_points, 3)

## Control points

Control points are 

## Multiscaling

Multiscaling allows to represent a shape at different scales ensuring consistency of landmarks, control points and signal accross scales. Read the documentation of Multiscaling to know more about this functionality.