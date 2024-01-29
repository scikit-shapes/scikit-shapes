:construction_worker: Scikit-shapes documentation is currently a work-in-progress, the more advances section of the doc is the [gallery of examples](/generated/gallery).

# Welcome to Scikit-Shapes documentation !

Scikit-shapes is a python library for shape analysis.

![](animation.gif)

### What is a shape ?

Loosely speaking, a shape is something that carry some geometric information. In a computer, a shape can be represented by two means:

- Polygonal structure: a point cloud (ie a collection of points in a 2D or 3D space) and eventually a structure of polygons. In scikit-shapes we are supporting three distinct types of polygonal data : points coulds, wireframe meshes, triangular meshes.
- Images: An image is a grid of pixels.

Typically, the shapes you may want to analyse with scikit-shapes are stored in some file format, for Polygonal Data it can be .ply, ... and for images .png, .nii...

For the moment, only polygonal structures are supported by scikit-shapes.

### What can I do with scikit-shapes ?

Scikit-shapes offers the possibility to get insight of shape data by various means.

- Single-shape level: compute statistics, compute features
- Multi-shape level: compute distances between two shapes or distances inside a population of shapes

Our goal is to make the integration of scikit-shapes into machine learning pipelines such as [scikit-learn](https://scikit-learn.org/stable/) seamless. You can think of scikit-shapes as a preprocessing tool that allows you to convert shape data (basically a format that ML tools are not able to read) to a description that can feed various ML/DL procedure. Scikit-shapes is intended to be your choice if you want to

- find clusters from a collection of 3d data based on morphological features
- compute a distance matrix for a set of poses
- perform landmark-aware alignment

### Where to start ?

First of all, you must install scikit-shapes, the easiest way is to install scikit-shapes and its dependencies via pip:
```bash
pip install skshapes
```
more in-depth coverage of installation can be foundin our [installation guide](installation.md)

Next, you can check out the [tutorials]() for having a taste of what is achievable with scikit-shapes. You can also find inspiration having a look at the [gallery of examples]().

If you are struggling using the package, feel free to open a [discussion]() on GitHub.

### Something is missing ? You find a bug ?

At the time you are reading these lines, scikit-shapes in under active development. We are open to contribution from the community. Could it be for a new feature, a bug fixing, a typo in the documentation or any kind of feedback, we warmly encourage you to open an issue on GitHub.
