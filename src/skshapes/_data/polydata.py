"""PolyData class."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Literal
from warnings import warn

import numpy as np
import pyvista
import torch
import vedo
from pyvista.core.pointset import PolyData as PyvistaPolyData

from ..cache import add_cached_methods_to_sphinx, cache_methods_and_properties
from ..errors import DeviceError, ShapeError
from ..input_validation import convert_inputs, one_and_only_one, typecheck
from ..multiscaling import Decimation, Multiscale
from ..triangle_mesh import EdgeTopology
from ..types import (
    Edges,
    EdgesLike,
    Float1dTensor,
    Int1dTensor,
    IntSequence,
    IntTensor,
    Landmarks,
    LandmarksSequence,
    Number,
    PointDensities,
    PointDensitiesLike,
    Points,
    PointsLike,
    Triangles,
    TrianglesLike,
    float_dtype,
    int_dtype,
    polydata_type,
)
from .data_attributes import DataAttributes


@add_cached_methods_to_sphinx
class PolyData(polydata_type):
    """A polygonal shape that is composed of points, edges and/or triangles.

    A :class:`PolyData` object can be created from:

    - Points, edges and triangles as `torch.Tensors <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_,
      `numpy.arrays <https://numpy.org/doc/stable/reference/generated/numpy.array.html>`_, lists or tuples of numbers.
    - A path to a file containing a mesh, in a format that is accepted by
      `pyvista.read <https://docs.pyvista.org/api/utilities/_autosummary/pyvista.read>`_.
    - A `pyvista.PolyData <https://docs.pyvista.org/api/core/_autosummary/pyvista.polydata>`_ object.
    - A `vedo.Mesh <https://vedo.embl.es/docs/vedo/mesh.html#Mesh>`_ object.

    .. warning::

        Internally, points are stored as **float32 torch Tensors** while indices for
        edges and triangles are stored as **int64 torch Tensors**. If you provide
        numpy arrays, lists or tuples, they will be converted to torch Tensors with the
        correct dtype. Since float32 precision corresponds to scientific notation
        with about 7 significant decimal digits, you may run into issues if you provide
        data that is centered far away from the origin.
        For instance, a sphere of radius 1 and centered at (1e6, 1e6, 1e6) will
        not be represented accurately. In such cases, we recommend centering the
        data around the origin before creating the :class:`PolyData` object.

        As a general rule, scikit-shapes accepts data in a wide range of formats
        but **always outputs torch Tensors** with float32 precision for point coordinates
        and int64 precision for indices.
        This design choice is motivated by a requirement for consistency
        (the output type of our methods should not surprise downstream functions)
        and support for both GPU computing and automatic differentiation
        (which are not supported by NumPy).


    Main features:

    - Custom scalar- or vector-valued **signals** on a :class:`PolyData` can
      be stored in the three attributes :attr:`point_data`,
      :attr:`edge_data` and :attr:`triangle_data` that behave as dictionaries of torch Tensors.

    - **Landmarks** or "keypoints" can also be defined on the shape,
      typically via a list of point indices. If :attr:`landmarks` is a sparse
      tensor of shape ``(n_landmarks, n_points)``, each row defines a landmark in
      barycentric coordinates. This allows us to define landmarks in a
      continuous space (e.g. on a triangle or an edge) instead
      of being constrained to the vertices of the shape.

    - For **visualization** purposes, a :class:`PolyData` can be converted to
      a `pyvista.PolyData <https://docs.pyvista.org/api/core/_autosummary/pyvista.polydata>`_
      or a `vedo.Mesh <https://vedo.embl.es/docs/vedo/mesh.html#Mesh>`_
      using the :meth:`to_pyvista` and :meth:`to_vedo` methods.

    - The :meth:`save` method lets you save shape data to any file whose format
      is supported by `PyVista <https://docs.pyvista.org/api/core/_autosummary/pyvista.polydata.save>`_ (.ply, .stl, .vtk...).


    Parameters
    ----------
    points
        The vertices of the shape, usually given as a
        ``(n_points, 3)`` torch Tensor of xyz coordinates for 3D data
        or as a ``(n_points, 2)`` torch Tensor of xy coordinates for 2D data.
        This first argument can also be a
        `vedo.Mesh <https://vedo.embl.es/docs/vedo/mesh.html#Mesh>`_,
        a `pyvista.PolyData <https://docs.pyvista.org/api/core/_autosummary/pyvista.polydata>`_ or a
        path to a file, in which case the edges and triangles are
        automatically inferred.
    edges
        The edges of the shape, understood as a set of non-oriented curves
        or as a wireframe mesh. The edges are encoded as a ``(n_edges, 2)`` torch Tensor
        of integer values. Each row ``[a, b]`` corresponds to a
        non-oriented edge between points ``a`` and ``b``.
        If ``triangles`` is not None, the edges will be automatically
        derived from it and this argument will be ignored.
    triangles
        The triangles of the shape, understood as a surface mesh.
        The triangles are encoded as a ``(n_triangles, 3)`` torch
        Tensor of integer values. Each row ``[a, b, c]`` corresponds to a
        triangle with vertices ``a``, ``b`` and ``c``.
        If ``edges`` and ``triangles`` are both None, the shape is
        understood as a point cloud.
    point_densities
        The densities of the points, which act as a multiplier to compute masses
        in :attr:`~skshapes.PolyData.point_masses`.
        For point clouds (with each point having a mass of 1 by default),
        point densities exactly correspond to point masses.
        For wireframes, point densities correspond to multipliers on the lengths
        of the edges.
        For triangle meshes, point densities correspond to multipliers on the areas
        of the triangles and can be understood as local thicknesses.
    device
        The device on which the shape is stored (e.g. ``"cpu"`` or ``"cuda"``).
        If None it is inferred from the points.
    landmarks
        The landmarks of the shape.
    control_points
        The control points of the shape.
    stiff_edges
        The stiff edges structure of the shape, useful for as_isometric_as_possible
    point_data
        The point data of the shape.
    edge_data
        The edge data of the shape.
    triangle_data
        The triangle data of the shape.
    cache_size
        Size of the cache for memoized properties. Defaults to None (= no
        cache limit). Use a smaller value if you intend to e.g. compute
        point curvatures at many different scales.

    Examples
    --------

    .. testcode::

        import skshapes as sks

        shape = sks.PolyData(
            points=[[0, 0], [1, 0], [0, 1], [1, 1]], triangles=[[0, 1, 2], [1, 2, 3]]
        )
        print(shape.points)


    .. testoutput::

        tensor([[0., 0.],
                [1., 0.],
                [0., 1.],
                [1., 1.]])

    .. testcode::

        print(shape.edges)

    .. testoutput::

        tensor([[0, 1],
                [0, 2],
                [1, 2],
                [1, 3],
                [2, 3]])

    .. testcode::

        print(shape.triangles)

    .. testoutput::

        tensor([[0, 1, 2],
                [1, 2, 3]])


    Please also check the :ref:`gallery <data_examples>`.

    """

    @convert_inputs
    @typecheck
    def __init__(
        self,
        points: PointsLike | vedo.Mesh | pyvista.PolyData | Path | str,
        *,
        edges: EdgesLike | None = None,
        triangles: TrianglesLike | None = None,
        point_densities: PointDensitiesLike | None = None,
        device: str | torch.device | None = None,
        landmarks: Landmarks | LandmarksSequence | None = None,
        control_points: PolyData | None = None,
        stiff_edges: EdgesLike | None = None,
        point_data: DataAttributes | None = None,
        edge_data: DataAttributes | None = None,
        triangle_data: DataAttributes | None = None,
        cache_size: int | None = None,
    ) -> None:
        # If the user provides a pyvista mesh, we extract the points, edges and
        # triangles from it
        # If the user provides a path to a mesh, we read it with pyvista and
        # extract the points, edges and triangles from it

        if not isinstance(points, torch.Tensor):
            if type(points) is vedo.Mesh:
                mesh = pyvista.PolyData(points.dataset)
            elif type(points) is PyvistaPolyData:
                mesh = points
            else:
                mesh = pyvista.read(points)

            # Now, mesh is a pyvista mesh
            cleaned_mesh = mesh.clean()

            if cleaned_mesh.n_points != mesh.n_points:
                mesh = cleaned_mesh
                if (
                    landmarks is not None
                    or "landmarks_indices" in mesh.field_data
                ):
                    warn(
                        "Mesh has been cleaned and points were removed."
                        + " Landmarks are ignored.",
                        stacklevel=3,
                    )
                    for i in mesh.field_data:
                        if i.startswith("landmarks"):
                            mesh.field_data.remove(i)
                    landmarks = None

                if point_data is not None or len(mesh.point_data) > 0:
                    warn(
                        "Mesh has been cleaned and points were removed."
                        + " point_data is ignored.",
                        stacklevel=3,
                    )
                    mesh.point_data.clear()
                    point_data = None

            # If the mesh is 2D, in pyvista it is a 3D mesh with z=0
            # We remove the z coordinate
            if np.allclose(mesh.points[:, 2], 0):
                points = torch.from_numpy(mesh.points[:, :2]).to(float_dtype)
            else:
                points = torch.from_numpy(mesh.points).to(float_dtype)

            if mesh.is_all_triangles:
                triangles = mesh.faces.reshape(-1, 4)[:, 1:]
                triangles = torch.from_numpy(triangles.copy())
                edges = None

            elif len(cleaned_mesh.faces) > 0 and len(cleaned_mesh.lines) > 0:
                raise ValueError(
                    "The mesh you try to convert to PolyData has both"
                    + " triangles and edges. This is not supported."
                )

            elif mesh.triangulate().is_all_triangles:
                triangles = mesh.triangulate().faces.reshape(-1, 4)[:, 1:]
                triangles = torch.from_numpy(triangles.copy())
                edges = None

            elif len(mesh.lines) > 0:
                edges = mesh.lines.reshape(-1, 3)[:, 1:]
                edges = torch.from_numpy(edges.copy())
                triangles = None

            elif len(cleaned_mesh.faces) == 0 and len(cleaned_mesh.lines) > 0:
                edges = cleaned_mesh.lines.reshape(-1, 3)[:, 1:]
                edges = torch.from_numpy(edges.copy())
                triangles = None

            else:
                edges = None
                triangles = None

            if len(mesh.point_data) > 0:
                point_data = DataAttributes.from_pyvista_datasetattributes(
                    mesh.point_data
                )
                for key in point_data:
                    if str(key) + "_shape" in mesh.field_data:
                        shape = tuple(mesh.field_data[str(key) + "_shape"])
                        point_data[key] = point_data[key].reshape(shape)

            if (
                ("landmarks_values" in mesh.field_data)
                and ("landmarks_indices" in mesh.field_data)
                and ("landmarks_size" in mesh.field_data)
            ):
                landmarks_from_pv = torch.sparse_coo_tensor(
                    values=mesh.field_data["landmarks_values"],
                    indices=mesh.field_data["landmarks_indices"],
                    size=tuple(mesh.field_data["landmarks_size"]),
                    dtype=float_dtype,
                )

            if any(x.startswith("edge_data") for x in mesh.field_data):
                edge_data = DataAttributes()
                prefix = "edge_data_"
                for key in mesh.field_data:
                    if key.startswith(prefix) and not key.endswith("_shape"):
                        shape = tuple(mesh.field_data[key + "_shape"])
                        edge_data[key[len(prefix) :]] = mesh.field_data[
                            key
                        ].reshape(shape)

            if any(x.startswith("triangle_data") for x in mesh.field_data):
                triangle_data = DataAttributes()
                prefix = "triangle_data_"
                for key in mesh.field_data:
                    if key.startswith(prefix) and not key.endswith("_shape"):
                        shape = tuple(mesh.field_data[key + "_shape"])
                        triangle_data[key[len(prefix) :]] = mesh.field_data[
                            key
                        ].reshape(shape)

        if device is None:
            device = points.device

        # We don't call the setters here because the setter of points is meant
        # to be used when the shape is modified in order to check the validity
        # of the new points
        self._points = points.clone().to(device)

        # /!\ If triangles is not None, edges will be ignored
        if triangles is not None:
            # Call the setter that will clone and check the validity of the triangles
            self.triangles = triangles
            # N.B.: we do not use the edges argument
            self._edges = None

        elif edges is not None:
            # Call the setter that will clone and check the validity of the
            # edges
            self.edges = edges
            self._triangles = None

        else:
            self._triangles = None
            self._edges = None

        self._point_densities = None
        if point_densities is not None:
            self.point_densities = point_densities

        if landmarks is not None:
            # Call the setter that will clone and check the validity of the
            # landmarks
            if hasattr(landmarks, "to_dense"):
                # landmarks is a sparse tensor
                self.landmarks = landmarks.to(self.device)
            else:
                # landmarks is a list of indices
                self.landmark_indices = landmarks

        elif "landmarks_from_pv" in locals():
            self.landmarks = landmarks_from_pv.to(self.device)
        else:
            self._landmarks = None

        if control_points is not None:
            self._control_points = control_points
        else:
            self._control_points = None

        self.device = device

        # Initialize data
        self.point_data = point_data
        self.edge_data = edge_data
        self.triangle_data = triangle_data

        # Initialize stiff_edges
        self._stiff_edges = stiff_edges

        # ----------------------------------------------------------------------------
        # Start of the Python magic (hack?) to load the features-computing methods,
        # memoize the properties (with cache clearing when users update the points,
        # edges or triangles), and add the methods to the class with a docstring that
        # is fully compatible with Sphinx autodoc.
        # ----------------------------------------------------------------------------

        # Cached methods: for reference on the Python syntax,
        # see "don't lru_cache methods! (intermediate) anthony explains #382",
        # https://www.youtube.com/watch?v=sVjtp6tGo0g
        cache_methods_and_properties(
            cls=PolyData,
            instance=self,
            cache_size=cache_size,
        )

    # N.B.: _cached_methods is also used in the decorator add_cached_methods_to_sphinx.
    _cached_methods = (
        "knn_graph",
        "k_ring_graph",
        "mesh_convolution",
        "point_convolution",
        "point_curvature_colors",
        "point_curvedness",
        "point_frames",
        "point_moments",
        "point_neighborhoods",
        "point_normals",
        "point_mean_gauss_curvatures",
        "point_principal_curvatures",
        "point_quadratic_coefficients",
        "point_quadratic_fits",
        "point_shape_indices",
    )
    _cached_properties = (
        "edge_lengths",
        "edge_midpoints",
        "edge_points",
        "point_masses",
        "triangle_area_normals",
        "triangle_areas",
        "triangle_centroids",
        "triangle_normals",
        "triangle_points",
    )

    # ----------------------------------------------------------------------------
    # End of the Python magic. Thanks StackOverflow and AnthonyExplains!
    # ----------------------------------------------------------------------------
    from .._features import (
        _edge_lengths,
        _edge_midpoints,
        _edge_points,
        _point_curvature_colors,
        _point_curvedness,
        _point_frames,
        _point_masses,
        _point_mean_gauss_curvatures,
        _point_moments,
        _point_normals,
        _point_principal_curvatures,
        _point_quadratic_coefficients,
        _point_quadratic_fits,
        _point_shape_indices,
        _triangle_area_normals,
        _triangle_areas,
        _triangle_centroids,
        _triangle_normals,
        _triangle_points,
    )
    from .._neighborhoods import _point_neighborhoods
    from ..cache import cache_clear
    from ..convolutions import _mesh_convolution, _point_convolution
    from ..topology import _k_ring_graph, _knn_graph

    @typecheck
    @one_and_only_one(["n_points", "ratio", "scale"])
    def resample(
        self,
        *,
        n_points: int | None = None,
        ratio: Number | None = None,
        scale: Number | None = None,
        method: Literal["auto"] = "auto",
        strict: bool = True,
    ) -> PolyData:
        """Resample the shape with a different number of vertices.

        .. warning::

            Supsampling has not been implemented yet.
            Currently, we only support decimation.

        To handle multiple scales at once, please consider using the
        :class:`Multiscale<skshapes.multiscaling.multiscale.Multiscale>` class.

        Parameters
        ----------
        n_points
            The number of points to keep in the output shape.
        ratio
            The ratio of points to keep in the output shape.
            A ratio of 1.0 keeps all the points, while a ratio of 0.1
            keeps 10% of the points.
        scale
            The typical distance between vertices in the output shape.
        method
            The method to use for resampling. Currently, only "auto" is
            supported. It corresponds to using quadratic decimation on
            a triangle surface mesh.
        strict
            If False, the decimation may run faster but with a value of n_points
            that is slightly different from the requested value.

        Raises
        ------
        InputStructureError
            If both target_reduction and n_points are provided.
            If none of target_reduction and n_points are provided.

        Returns
        -------
        PolyData
            The decimated shape.

        Examples
        --------

        .. testcode::

            import skshapes as sks

            shape = sks.Sphere()
            lowres = shape.resample(ratio=0.1)
            print(f"from {shape.n_points} to {lowres.n_points} points")

        .. testoutput::

            from 842 to 84 points

        .. testcode::

            lowres = shape.resample(n_points=100)
            print(f"from {shape.n_points} to {lowres.n_points} points")

        .. testoutput::

            from 842 to 100 points

        .. testcode::

            lowres = shape.resample(n_points=100, strict=False)
            print(f"from {shape.n_points} to {lowres.n_points} points")

        .. testoutput::

            from 842 to 101 points

        """

        if scale is not None:
            msg = "Resampling with a scale is not implemented yet."
            raise NotImplementedError(msg)

        if method != "auto":
            msg = "Only the 'auto' method is implemented for now."
            raise NotImplementedError(msg)

        if self.is_triangle_mesh:
            kwargs = {
                "n_points": n_points,
                "ratio": ratio,
            }
            if strict:
                multi_self = Multiscale(self, ratios=[1])
                multi_self.append(**kwargs)
                return multi_self.at(**kwargs)

            d = Decimation(**kwargs)
            return d.fit_transform(self)

        else:
            msg = "Decimation is only implemented for triangle meshes so far."
            raise NotImplementedError(msg)

    @typecheck
    def save(self, filename: Path | str) -> None:
        """Save the shape to a file.

        Format accepted by PyVista are supported (.ply, .stl, .vtk)
        see: https://github.com/pyvista/pyvista/blob/release/0.40/pyvista/core/pointset.py#L439-L1283  # noqa: E501

        Parameters
        ----------
        filename
            The path where to save the shape.
        """
        mesh = self.to_pyvista()
        mesh.save(filename)

    #########################
    #### Copy functions #####
    #########################
    @typecheck
    def copy(self) -> PolyData:
        """Copy the shape.

        Returns
        -------
        PolyData
            The copy of the shape.


        Examples
        --------

        .. testcode::

            import skshapes as sks

            a = sks.Sphere()
            b = a.copy()
            b.points[1, :] = 7

            print("Original:")
            print(a.points[0:4, :])
            print("Edited copy:")
            print(b.points[0:4, :])

        .. testoutput::

            Original:
            tensor([[ 0.0000,  0.0000,  0.5000],
                    [ 0.0000,  0.0000, -0.5000],
                    [ 0.0541,  0.0000,  0.4971],
                    [ 0.1075,  0.0000,  0.4883]])
            Edited copy:
            tensor([[0.0000, 0.0000, 0.5000],
                    [7.0000, 7.0000, 7.0000],
                    [0.0541, 0.0000, 0.4971],
                    [0.1075, 0.0000, 0.4883]])

        """
        kwargs = {"points": self._points.clone()}

        if self._triangles is not None:
            kwargs["triangles"] = self._triangles.clone()
        if self._edges is not None:
            kwargs["edges"] = self._edges.clone()
        if self._landmarks is not None:
            kwargs["landmarks"] = self._landmarks.clone()
        if self._point_data is not None:
            kwargs["point_data"] = self._point_data.clone()
        if self._edge_data is not None:
            kwargs["edge_data"] = self._edge_data.clone()
        if self._triangle_data is not None:
            kwargs["triangle_data"] = self._triangle_data.clone()
        if self._control_points is not None:
            kwargs["control_points"] = self.control_points.copy()
        if self._stiff_edges is not None:
            kwargs["stiff_edges"] = self.stiff_edges.clone()

        return PolyData(**kwargs, device=self.device)

    @typecheck
    def to(self, device: str | torch.device) -> PolyData:
        """Copy the shape onto a given device."""
        torch_device = torch.Tensor().to(device).device
        if self.device == torch_device:
            return self
        else:
            copy = self.copy()
            copy.device = device
            return copy

    ###########################
    #### Vedo interface #######
    ###########################
    @typecheck
    def to_vedo(self) -> vedo.Mesh:
        """Converts the shape to a `vedo.Mesh <https://vedo.embl.es/docs/vedo/mesh.html#Mesh>`_ object.

        Returns
        -------
        vedo.Mesh
            The shape as a `vedo.Mesh <https://vedo.embl.es/docs/vedo/mesh.html#Mesh>`_ object.


        Examples
        --------

        .. code-block:: python

            import skshapes as sks

            shape = sks.Sphere()
            print(shape.to_vedo())

        .. code-block::

            vedo.mesh.Mesh at (0x30e7f0e0)
            name          : Mesh
            elements      : vertices=842 polygons=1,680 lines=0
            position      : (0, 0, 0)
            scaling       : (1.00000, 1.00000, 1.00000)
            size          : average=0.500000, diagonal=1.72721
            center of mass: (0, 0, 0)
            bounds        : x=(-0.499, 0.499), y=(-0.497, 0.497), z=(-0.500, 0.500)

        """
        return vedo.Mesh(self.to_pyvista())

    ###########################
    #### PyVista interface ####
    ###########################
    @typecheck
    def to_pyvista(self) -> pyvista.PolyData:
        """Converts the shape to a `pyvista.PolyData <https://docs.pyvista.org/api/core/_autosummary/pyvista.polydata>`_ object.

        Returns
        -------
        pyvista.PolyData
            The shape as a `pyvista.PolyData <https://docs.pyvista.org/api/core/_autosummary/pyvista.polydata>`_ object.

        Examples
        --------

        .. code-block:: python

            import skshapes as sks

            shape = sks.Sphere()
            print(shape.to_pyvista())

        .. code-block::

            PolyData (0x7f9b96ab0e80)
            N Cells:    1680
            N Points:   842
            N Strips:   0
            X Bounds:   -4.993e-01, 4.993e-01
            Y Bounds:   -4.965e-01, 4.965e-01
            Z Bounds:   -5.000e-01, 5.000e-01
            N Arrays:   0

        """
        if self.dim == 3:
            points = self.points.detach().cpu().numpy()
        else:
            points = np.concatenate(
                [
                    self.points.detach().cpu().numpy(),
                    np.zeros((self.n_points, 1)),
                ],
                axis=1,
            )

        if self._triangles is not None:
            np_triangles = self._triangles.detach().cpu().numpy()
            faces = np.concatenate(
                [
                    np.ones((self.n_triangles, 1), dtype=np.int64) * 3,
                    np_triangles,
                ],
                axis=1,
            )
            polydata = pyvista.PolyData(points, faces=faces)

        elif self._edges is not None:
            np_edges = self._edges.detach().cpu().numpy()
            lines = np.concatenate(
                [np.ones((self.n_edges, 1), dtype=np.int64) * 2, np_edges],
                axis=1,
            )
            polydata = pyvista.PolyData(points, lines=lines)

        else:
            polydata = pyvista.PolyData(points)

        # Add the point data if any
        if len(self.point_data) > 0:
            point_data_dict = self.point_data.to_numpy_dict()
            for key in point_data_dict:
                if len(point_data_dict[key].shape) <= 2:
                    # If the data is 1D or 2D, we add it as a point data
                    polydata.point_data[key] = point_data_dict[key]
                else:
                    # If the data is 3D or more, we must be careful
                    # because pyvista does not support 3D or more point data
                    polydata.point_data[key] = point_data_dict[key].reshape(
                        self.n_points, -1
                    )
                    polydata.field_data[str(key) + "_shape"] = point_data_dict[
                        key
                    ].shape

        # Add edge_data as field_data
        if len(self.edge_data) > 0:
            edge_data_dict = self.edge_data.to_numpy_dict()
            for key in edge_data_dict:
                shape = edge_data_dict[key].shape
                flat = edge_data_dict[key].reshape(self.n_edges, -1)
                polydata.field_data["edge_data_" + str(key)] = flat
                polydata.field_data["edge_data_" + str(key) + "_shape"] = shape

        # Add triangle_data as field_data
        if len(self.triangle_data) > 0:
            triangle_data_dict = self.triangle_data.to_numpy_dict()
            for key in triangle_data_dict:
                shape = triangle_data_dict[key].shape
                flat = triangle_data_dict[key].reshape(self.n_triangles, -1)
                polydata.field_data["triangle_data_" + str(key)] = flat
                polydata.field_data["triangle_data_" + str(key) + "_shape"] = (
                    shape
                )

        # Add the landmarks if any
        if hasattr(self, "_landmarks") and self.landmarks is not None:
            coalesced_landmarks = self.landmarks.coalesce()
            polydata.field_data["landmarks_values"] = (
                coalesced_landmarks.values()
            )
            polydata.field_data["landmarks_indices"] = (
                coalesced_landmarks.indices()
            )
            polydata.field_data["landmarks_size"] = coalesced_landmarks.size()
            polydata.field_data["landmark_points"] = (
                self.landmark_points.detach()
            )

        return polydata

    ###########################
    #### Plotting utilities ###
    ###########################

    def plot(
        self,
        backend: Literal["pyvista", "vedo"] = "pyvista",
        **kwargs,
    ) -> None:
        """Displays the shape, typically using a PyVista interactive window.

        Available backends are ``"pyvista"`` and ``"vedo"``. See the documentation of
        the corresponding plot methods for possible arguments:

        - `pyvista <https://docs.pyvista.org/version/stable/api/core/_autosummary/pyvista.PointSet.plot.html>`__
        - `vedo <https://vedo.embl.es/docs/vedo/plotter.html#show>`__


        Parameters
        ----------
        backend
            Which backend to use for plotting.
        """
        if backend == "pyvista":
            self.to_pyvista().plot(**kwargs)

        elif backend == "vedo":
            self.to_vedo().show(**kwargs)

    #############################
    #### Edges getter/setter ####
    #############################
    @property
    @typecheck
    def edges(self) -> Edges | None:
        """The edges of the shape.

        If the shape is a triangle mesh, the edges
        are computed from the triangles. If the shape is not a triangle
        mesh, the edges are directly returned.
        If the shape is a point cloud without explicit topology information,
        returns ``None``.
        """
        if self._edges is not None:
            return self._edges

        elif self._triangles is not None:
            edges = EdgeTopology(self._triangles).edges
            self._edges = edges
            return edges

        return None

    @edges.setter
    @convert_inputs
    @typecheck
    def edges(self, edges: Edges | None) -> None:
        """Set the edges of the shape and set the triangles to None.

        Also reinitialize edge_data, triangle_data and the cache.
        """
        if edges is None:
            self._edges = None
        else:
            if edges.max() >= self.n_points:
                raise IndexError(
                    "The maximum vertex index in edges array is larger than the"
                    + " number of points."
                )
            self._edges = edges.clone().to(self.device)
        self._triangles = None
        self.edge_data = None
        self.triangle_data = None
        self.cache_clear()

    ###################################
    #### Stiff Edges getter/setter ####
    ###################################
    @property
    @typecheck
    def stiff_edges(self) -> Edges | None:
        """Stiff edges getter"""

        if not hasattr(self, "_stiff_edges"):
            return None

        return self._stiff_edges

    @stiff_edges.setter
    @typecheck
    def stiff_edges(self, stiff_edges: Edges | None) -> None:
        """Set the stiff edges of the PolyData

        Stiff edges are additional edges that can used as edge structure for
        some some intrinsic metrics such as as_isometric_as_possible.

        Built-in methods such as `k_ring_graph` or `knn_graph` can be used
        to compute the stiff edges.

        Parameters
        ----------
        stiff_edges
            The stiff edges

        Raises
        ------
        IndexError
            If the maximum index in stiff_edges exceeds the number of points
        """

        if stiff_edges is None:
            self._stiff_edges = None

        else:
            if stiff_edges.max() >= self.n_points:
                raise IndexError(
                    "The maximum vertex index in edges array is larger than the"
                    + " number of points."
                )

            self._stiff_edges = stiff_edges.clone().to(self.device)

    #################################
    #### Triangles getter/setter ####
    #################################
    @property
    @typecheck
    def triangles(self) -> Triangles | None:
        """Triangles getter."""
        return self._triangles

    @triangles.setter
    @convert_inputs
    @typecheck
    def triangles(self, triangles: Triangles | None) -> None:
        """Set the triangles of the shape and set edges to None.

        Also reinitialize edge_data, triangle_data and the cache.
        """
        if triangles is None:
            self._triangles = None
        else:
            if triangles.max() >= self.n_points:
                raise IndexError(
                    "The maximum vertex index in triangles array is larger than"
                    + " the number of points."
                )

            self._triangles = triangles.clone().to(self.device)

        self._edges = None
        self.edge_data = None
        self.triangle_data = None
        self.cache_clear()

    ##############################
    #### Points getter/setter ####
    ##############################
    @property
    @typecheck
    def points(self) -> Points:
        """Points getter."""
        return self._points

    @points.setter
    @convert_inputs
    @typecheck
    def points(self, points: Points) -> None:
        """Points setter.

        Parameters
        ----------
        points
            The new points of the shape.

        Raises
        ------
        ShapeError
            If the new number of points is different from the actual number of
            points in the shape.
        """
        if points.shape[0] != self.n_points:
            msg = "The number of points cannot be changed."
            raise ShapeError(msg)

        self._points = points.clone().to(self.device)
        self.cache_clear()

    #######################################
    #### Point Densities setter/getter ####
    #######################################
    @property
    @typecheck
    def point_densities(self) -> PointDensities:
        """The density of each point of the shape.

        The densities of the points act as a multiplier to compute masses
        in :attr:`~skshapes.PolyData.point_masses`.
        For point clouds (with each point having a mass of 1 by default),
        point densities exactly correspond to point masses.
        For wireframes, point densities correspond to multipliers on the lengths
        of the edges.
        For triangle meshes, point densities correspond to multipliers on the areas
        of the triangles and can be understood as local thicknesses.
        """
        # TODO: add support for multiscaling
        if self._point_densities is None:
            self._point_densities = torch.ones_like(self.points[:, 0])
        return self._point_densities

    @point_densities.setter
    @convert_inputs
    @typecheck
    def point_densities(self, densities: PointDensities) -> None:
        """Setter for the point densities.

        Parameters
        ----------
        densities
            The new densities for the points of the shape.

        Raises
        ------
        ShapeError
            If the new density is not a vector of shape ``(n_points,)``.
        """
        if densities.shape != (self.n_points,):
            msg = f"The new densities should be a vector of shape (n_points,) = ({self.n_points},)."
            raise ShapeError(msg)

        self._point_densities = densities.clone().to(self.device)
        # Point densities play an important role in convolutions
        # and neighborhood computations, so we clear the cache
        self.cache_clear()

    ##############################
    #### Device getter/setter ####
    ##############################
    @property
    @typecheck
    def device(self) -> torch.device:
        """Device getter."""
        return self.points.device

    @device.setter
    @typecheck
    def device(self, device: str | torch.device) -> None:
        """Device setter.

        Parameters
        ----------
        device
            The device on which the shape should be stored.
        """
        for attr in dir(self):
            if attr.startswith("_"):
                attribute = getattr(self, attr)
                if isinstance(attribute, torch.Tensor):
                    setattr(self, attr, attribute.to(device))

        if hasattr(self, "_point_data") and self._point_data is not None:
            self._point_data = self._point_data.to(device)
        if hasattr(self, "_edge_data") and self._edge_data is not None:
            self._edge_data = self._edge_data.to(device)
        if hasattr(self, "_triangle_data") and self._triangle_data is not None:
            self._triangle_data = self._triangle_data.to(device)
        if (
            hasattr(self, "_control_points")
            and self._control_points is not None
        ):
            self._control_points = self._control_points.to(device)

    ################################
    #### point_data getter/setter ##
    ################################
    @property
    @typecheck
    def point_data(self) -> DataAttributes:
        """Point data getter."""
        return self._point_data

    @point_data.setter
    @typecheck
    def point_data(self, point_data: DataAttributes | dict | None) -> None:
        """Point data setter.

        Parameters
        ----------
        point_data_dict
            The new point data of the shape.
        """
        self._point_data = self._generic_data_attribute_setter(
            value=point_data, n=self.n_points, name="point_data"
        )

    ################################
    #### edge_data getter/setter ###
    ################################

    @property
    @typecheck
    def edge_data(self) -> DataAttributes:
        """Edge data getter."""
        return self._edge_data

    @edge_data.setter
    @typecheck
    def edge_data(self, edge_data: DataAttributes | dict | None) -> None:
        """Edge data setter.

        Parameters
        ----------
        edge_data
            The new edge data of the shape.
        """
        self._edge_data = self._generic_data_attribute_setter(
            value=edge_data, n=self.n_edges, name="edge_data"
        )

    ####################################
    #### triangle_data getter/setter ###
    ####################################

    @property
    @typecheck
    def triangle_data(self) -> DataAttributes:
        """Triangle data getter."""
        return self._triangle_data

    @triangle_data.setter
    @typecheck
    def triangle_data(
        self, triangle_data: DataAttributes | dict | None
    ) -> None:
        """Triangle data setter.

        Parameters
        ----------
        triangle_data
            The new triangle data of the shape.
        """
        self._triangle_data = self._generic_data_attribute_setter(
            value=triangle_data, n=self.n_triangles, name="triangle_data"
        )

    def _generic_data_attribute_setter(
        self,
        value: DataAttributes | dict | None,
        n: int,
        name: str,
    ) -> DataAttributes:
        """Generic data setter.

        Reusable method to set the point_data, edge_data and triangle_data

        Parameters
        ----------
        value
            The data initializer, can be a DataAttributes, a dictionary or None.
        n
            The expected first dimension of the data (typically the number of
            points, edges or triangles).
        name
            The name of the attribute (point_data, edge_data or triangle_data).
            Used for error messages.

        Returns
        -------
        DataAttributes
            The data attributes.

        Raises
        ------
        ShapeError
            If the first dimension of the data is different from the
            expected one.
        """

        if value is None:
            data_attr = DataAttributes(n=n, device=self.device)
        elif not isinstance(value, DataAttributes):
            data_attr = DataAttributes.from_dict(value).to(self.device)
        else:
            data_attr = value.to(self.device)

        if data_attr.n != n:
            error_message = (
                f"The expected first dimension of {name} entries should be {n}"
                + f" but got {data_attr.n}."
            )
            raise ShapeError(error_message)
        return data_attr

    #################################
    #### Landmarks getter/setter ####
    #################################
    @property
    @typecheck
    def landmarks(self) -> Landmarks | None:
        """Get the landmarks of the shape.

        The format is a sparse tensor of shape (n_landmarks, n_points), each
        line is a landmark in barycentric coordinates. If you want to get the
        landmarks in 3D coordinates, use the landmark_points property. If you
        want to get the landmarks as a list of indices, use the
        landmark_indices property.

        If no landmarks are defined, returns None.
        """
        return self._landmarks

    @property
    @typecheck
    def n_landmarks(self) -> int:
        """Return the number of landmarks."""
        if self.landmarks is None:
            return 0
        else:
            return self.landmarks.shape[0]

    @landmarks.setter
    @convert_inputs
    @typecheck
    def landmarks(self, landmarks: Landmarks) -> None:
        """Landmarks setter.

        The landmarks should be a sparse tensor of shape
        (n_landmarks, n_points) (barycentric coordinates).
        """
        assert landmarks.is_sparse
        assert landmarks.shape[1] == self.n_points
        assert landmarks.dtype == float_dtype
        self._landmarks = landmarks.clone().to(self.device)

    @property
    @typecheck
    def landmark_points(self) -> Points | None:
        """Landmarks in spatial coordinates."""
        if self.landmarks is None:
            return None
        else:
            return self.landmarks @ self.points

    @property
    def landmark_points_3D(self) -> Points | None:
        """Landmarks in 3D coordinates.

        If self.dim == 3, it is equivalent to landmark_points. Otherwise, if
        self.dim == 2, it returns the landmarks with a third coordinate set to 0.

        Returns
        -------
        Points
            The landmarks in 3D coordinates.
        """
        if self.dim == 3:
            return self.landmark_points
        else:
            landmark_points = torch.zeros(
                (self.n_landmarks, 3), dtype=float_dtype, device=self.device
            )
            landmark_points[:, : self.dim] = self.landmark_points
            return landmark_points

    @property
    @typecheck
    def landmark_indices(self) -> IntTensor | None:
        """Indices of the landmarks.

        Raises
        ------
        ValueError
            If the landmarks are not indices (there are defined in barycentric
            coordinates).

        Returns
        -------
        Optional[IntTensor]
            The indices of the landmarks. None if no landmarks are defined.

        """
        if self.landmarks is None:
            return None
        else:
            coalesced_landmarks = self.landmarks.coalesce()
            values = coalesced_landmarks.values()
            indices = coalesced_landmarks.indices()[1][values == 1]

            if len(indices) != self.n_landmarks:
                msg = "Landmarks are not indices."
                raise ValueError(msg)

            return indices[values == 1]

    @landmark_indices.setter
    @convert_inputs
    @typecheck
    def landmark_indices(self, landmarks: Int1dTensor | list[int]) -> None:
        """Landmarks setter with indices list.

        Parameters
        ----------
        landmarks
            The indices of the landmarks.

        """
        if isinstance(landmarks, list):
            landmarks = torch.tensor(landmarks, dtype=int_dtype)

        assert landmarks.max() < self.n_points

        n_landmarks = len(landmarks)
        n_points = self.n_points

        indices = torch.zeros((2, n_landmarks), dtype=int_dtype)
        indices[0] = torch.arange(n_landmarks, dtype=int_dtype)
        indices[1] = landmarks

        values = torch.ones_like(indices[0], dtype=float_dtype)

        self.landmarks = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(n_landmarks, n_points),
            device=self.device,
        )

    def add_landmarks(self, indices: IntSequence | int) -> None:
        """Add vertices landmarks to the shape.

        Parameters
        ----------
        indices
            The indices of the vertices to add to the landmarks.
        """
        if not hasattr(indices, "__iter__"):
            self.add_landmarks([indices])

        elif self.landmarks is None:
            self.landmark_indices = indices

        else:
            new_indices = torch.tensor(
                indices, dtype=int_dtype, device=self.device
            )

            coalesced_landmarks = self.landmarks.coalesce()
            old_values = coalesced_landmarks.values()
            old_indices = coalesced_landmarks.indices()

            n_new_landmarks = len(new_indices)
            new_indices = torch.zeros(
                (2, n_new_landmarks), dtype=int_dtype, device=self.device
            )
            new_indices[0] = (
                torch.arange(n_new_landmarks, dtype=int_dtype)
                + self.n_landmarks
            )
            new_indices[1] = torch.tensor(
                indices, dtype=int_dtype, device=self.device
            )

            new_values = torch.ones_like(
                new_indices[0], dtype=float_dtype, device=self.device
            )

            n_landmarks = self.n_landmarks + n_new_landmarks

            indices = torch.cat((old_indices, new_indices), dim=1)
            values = torch.concat((old_values, new_values))

            self.landmarks = torch.sparse_coo_tensor(
                indices=indices,
                values=values,
                size=(n_landmarks, self.n_points),
                device=self.device,
            )

    ##########################
    #### Shape properties ####
    ##########################
    @property
    @typecheck
    def dim(self) -> int:
        """Dimension of the shape getter."""
        return self.points.shape[1]

    @property
    @typecheck
    def n_points(self) -> int:
        """Number of points getter."""
        return self.points.shape[0]

    @property
    @typecheck
    def n_edges(self) -> int:
        """Number of edges getter."""
        edges = self.edges
        if edges is not None:
            return edges.shape[0]
        else:
            return 0

    @property
    @typecheck
    def n_triangles(self) -> int:
        """Number of triangles getter."""
        if self._triangles is not None:
            return self._triangles.shape[0]
        else:
            return 0

    @cached_property
    @typecheck
    def mean_point(self) -> Points:
        """Mean point of the shape.

        Return the mean point as a (N_batch, 3) tensor.
        """
        # TODO: add support for batch vectors
        # TODO: add support for point weights
        return self.points.mean(dim=0, keepdim=True)

    @cached_property
    @typecheck
    def standard_deviation(self) -> Float1dTensor:
        """Standard deviation of the shape.

        Returns the standard deviation (radius) of the shape as a (N_batch,)
        tensor.
        """
        # TODO: add support for batch vectors
        # TODO: add support for point weights
        return (
            ((self.points - self.mean_point) ** 2)
            .sum(dim=1)
            .mean(dim=0)
            .sqrt()
            .view(-1)
        )

    @cached_property
    @typecheck
    def radius(self) -> Float1dTensor:
        """Radius of the shape.

        Returns the radius of the smallest sphere, centered around the mean point, that contains the shape.
        """
        # TODO: add support for batch vectors
        return (
            ((self.points - self.mean_point) ** 2)
            .sum(dim=1)
            .max(dim=0)
            .values.sqrt()
            .view(-1)
        )

    @typecheck
    def normalize(self, inplace=False) -> PolyData:
        """Normalize the shape.

        Center the shape at the origin and scale it so that the standard
        deviation of the points is 1.

        Returns
        -------
        PolyData
            The normalized shape.
        """
        new = self if inplace else self.copy()

        new.points = new.points - new.mean_point
        new.points = new.points / new.radius

        return new

    @property
    @typecheck
    def is_triangle_mesh(self) -> bool:
        """Check if the shape has triangle connectivity."""
        return self._triangles is not None

    @property
    @typecheck
    def is_wireframe(self) -> bool:
        """Check if the shape has edge connectivity."""
        return (self._triangles is None) and (self._edges is not None)

    @property
    @typecheck
    def is_point_cloud(self) -> bool:
        """Check if the shape is a point cloud."""
        return (self._triangles is None) and (self._edges is None)

    @typecheck
    def bounding_grid(
        self, N: int = 10, offset: float | int = 0.05
    ) -> polydata_type:
        """Bounding grid of the shape.

        Compute a bounding grid of the shape. The grid is a PolyData with
        points and edges representing a regular grid enclosing the shape.

        Parameters
        ----------
        N
            The number of points on each axis of the grid.
        offset
            The offset of the grid with respect to the shape. If offset=0, the
            grid is exactly the bounding box of the shape. If offset=1, the
            grid is the bounding box of the shape dilated by a factor 2.

        Returns
        -------
        PolyData
            The bounding grid of the shape.
        """
        pv_shape = self.to_pyvista()
        # Get the bounds of the mesh
        xmin, xmax, ymin, ymax, zmin, zmax = pv_shape.bounds
        spacing = (
            (1 + 2 * offset) * (xmax - xmin) / (N - 1),
            (1 + 2 * offset) * (ymax - ymin) / (N - 1),
            (1 + 2 * offset) * (zmax - zmin) / (N - 1),
        )
        origin = (
            xmin - offset * (xmax - xmin),
            ymin - offset * (ymax - ymin),
            zmin - offset * (zmax - zmin),
        )

        # Create the grid
        grid = pyvista.ImageData(
            dimensions=(N, N, N),
            spacing=spacing,
            origin=origin,
        ).extract_all_edges()  # Extract the grid as wireframe mesh

        return PolyData(grid)

    @property
    @typecheck
    def control_points(self) -> PolyData | None:
        """Control points of the shape."""
        return self._control_points

    @control_points.setter
    @convert_inputs
    @typecheck
    def control_points(self, control_points: polydata_type | None) -> None:
        """Set the control points of the shape.

        The control points are a PolyData object. The principal use case of
        control points is in [`ExtrinsicDeformation`][skshapes.morphing.extrinsic_deformation.ExtrinsicDeformation] # noqa: E501

        Parameters
        ----------
        control_points
            PolyData representing the control points.

        Raises
        ------
        DeviceError
            If `self.device != control_points.device`.

        """
        if control_points is not None and self.device != control_points.device:
            raise DeviceError(
                "Controls points must be on the same device as"
                + " the corresponding PolyData, found "
                + f"{control_points.device} for control points"
                + f" and {self.device} for the PolyData."
            )
        self._control_points = control_points

    ###########################
    #### Weighted point cloud #
    ###########################

    @typecheck
    def to_weighted_points(self) -> tuple[Points, Float1dTensor]:
        """Convert the shape to a weighted point cloud.

        Returns
        -------
        points
            The points of the weighted point cloud.
        weights
            The weights of the weighted point cloud.
        """
        if self.is_triangle_mesh:
            points = self.triangle_centroids
            weights = self.triangle_areas / self.triangle_areas.sum()

        elif self.is_wireframe:
            points = self.edge_midpoints
            weights = self.edge_lengths / self.edge_lengths.sum()

        else:
            points = self.points
            weights = (
                torch.ones(
                    self.n_points, dtype=float_dtype, device=self.device
                )
                / self.n_points
            )

        return points, weights

    @typecheck
    def to_point_cloud(self) -> PolyData:
        new = self.copy()
        new.edges = None
        new.triangles = None
        return new

    @typecheck
    def __repr__(self) -> str:
        """String representation of the shape.

        Returns
        -------
        str
            A succinct description of the shape data.

        Examples
        --------

        .. testcode::

            import skshapes as sks

            shape = sks.Sphere()
            print(shape)

        .. testoutput::

            skshapes.PolyData (... on cpu, float32), a 3D triangle mesh with:
            - 842 points, 2,520 edges, 1,680 triangles
            - center (-0.000, -0.000, -0.000), radius 0.500
            - bounds x=(-0.499, 0.499), y=(-0.497, 0.497), z=(-0.500, 0.500)

        .. testcode::

            shape.point_data["norms"] = shape.points.norm(dim=1)
            shape.edge_data["squared_indices"] = shape.edges**2
            tri = shape.triangles
            shape.triangle_data["tensors"] = tri.view(-1, 3, 1) * tri.view(-1, 1, 3)
            print(shape)

        .. testoutput::

            skshapes.PolyData (... on cpu, float32), a 3D triangle mesh with:
            - 842 points, 2,520 edges, 1,680 triangles
            - center (-0.000, -0.000, -0.000), radius 0.500
            - bounds x=(-0.499, 0.499), y=(-0.497, 0.497), z=(-0.500, 0.500)
            - point data:
              : "norms" : [842], float32, cpu
            - edge data:
              : "squared_indices" : [2520, 2], int64, cpu
            - triangle data:
              : "tensors" : [1680, 3, 3], int64, cpu

        """

        if self.is_triangle_mesh:
            shape_type = "triangle mesh"
            info = f"{self.n_points:,} points, {self.n_edges:,} edges, {self.n_triangles:,} triangles"
        elif self.is_wireframe:
            shape_type = "wireframe"
            info = f"{self.n_points:,} points and {self.n_edges:,} edges"
        elif self.is_point_cloud:
            shape_type = "point cloud"
            info = f"{self.n_points:,} points"
        else:
            msg = "Unknown shape type"
            raise NotImplementedError(msg)

        def suffix(dtype):
            return str(dtype).split(".")[-1].split("'")[0]

        repr_str = f"skshapes.{self.__class__.__name__} (0x{id(self):x} on {self.device}, {suffix(float_dtype)}), "
        repr_str += f"a {self.dim}D {shape_type} with:\n"
        repr_str += f"- {info}\n"

        center = ", ".join(
            f"{coord:.3f}"
            for coord in self.mean_point[0].detach().cpu().numpy()
        )
        repr_str += (
            f"- center ({center}), radius {float(self.radius.item()):.3f}\n"
        )
        repr_str += f"- bounds x=({self.points[:, 0].min():.3f}, {self.points[:, 0].max():.3f}), "
        repr_str += f"y=({self.points[:, 1].min():.3f}, {self.points[:, 1].max():.3f}), "
        if self.dim == 3:
            repr_str += f"z=({self.points[:, 2].min():.3f}, {self.points[:, 2].max():.3f})"

        for data, data_name in [
            (self.point_data, "point data"),
            (self.edge_data, "edge data"),
            (self.triangle_data, "triangle data"),
        ]:
            if len(data) > 0:
                repr_str += f"\n- {data_name}:"
                for key in data:
                    repr_str += f'\n  : "{key}" : {list(data[key].shape)}, {suffix(data[key].dtype)}, {data.device}'

        return repr_str
