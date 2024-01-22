# Welcome to Scikit-Shapes documentation !

Scikit-shapes is a python library for shape analysis.

![](animation.gif)

### What is a shape ?

Loosely speaking, a shape is something that carry some geometric information. In a computer, a shape can be represented by two means:

- Polygonal structure: a point cloud (ie a collection of points in a 2D or 3D space) and eventually a structure of polygons. In scikit-shapes we are supporting three distinct types of polygonal data : points coulds, wireframe meshes, triangular meshes.
- Images: An image is a grid of pixels.

Typically, the shapes you may want to analyse with scikit-shapes are stored in some file format, for Polygonal Data it can be .ply, ... and for images .png, .nii...

Want to go deeper on the differences between polygonal data and images and their implication for computation ? Check out the section [what makes PolyData and Images different]() of the [scikit-shapes guide]().

### What can I do with scikit-shapes ?

Scikit-shapes offers the possibility to get insight of shape data by various means.

- Single-shape level: compute statistics, compute features
- Multi-shape level: compute distances between two shapes or distances inside a population of shapes

Our goal is to make the integration of scikit-shapes into machine learning pipelines such as [scikit-learn](https://scikit-learn.org/stable/) seamless. You can think of scikit-shapes as a preprocessing tool that allows you to convert shape data (basically a format that ML tools are not able to read) to a descritpion that can feed various ML/DL procedure. One typical situation is the following:

- You have a collection of 3d data that you may want to clusterized
- You can use scikit-shapes to compute a distance matrix
- You can use this distance matrix to do a clustering of the initial data thanks to scikit-learn

### Where to start ?

First of all, you must install scikit-shapes, the easiest way is to install scikit-shapes and its dependencies via pip:
```bash
pip install skshapes
```
more in-depth coverage of installation can be foundin our [installation guide](installation.md)

Next, you can check out the [scikit-shapes guide]() where you can find information about the algorithms and the features of scikit-shapes. You can also find inspiration having a look at the [gallery of examples](). If you are struggling using the package, feel free to open a [discussion]() on GitHub.

### Something is missing ? You find a bug ?

At the time you are reading these lines, scikit-shapes in under active development. We are open to contribution from the community. If you want to contribute, with a new feature, a bug spotting, or with any feedback, we will greatly appreciate your help. Have a look at our [contribution guide]() and reach us on [Github]()
