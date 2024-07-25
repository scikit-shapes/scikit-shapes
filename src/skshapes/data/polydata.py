"""PolyData class."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal
from warnings import warn

import numpy as np
import pyvista
import torch
import vedo
from pyvista.core.pointset import PolyData as PyvistaPolyData

from ..errors import DeviceError, ShapeError
from ..input_validation import convert_inputs, one_and_only_one, typecheck
from ..triangle_mesh import EdgeTopology
from ..types import (
    Edges,
    Float1dTensor,
    Float2dTensor,
    Int1dTensor,
    IntSequence,
    IntTensor,
    Landmarks,
    Number,
    Points,
    Triangles,
    float_dtype,
    int_dtype,
    polydata_type,
)
from .utils import DataAttributes


class PolyData(polydata_type):
    """A polygonal data object. It is composed of points, edges and triangles.

    Three types of objects can be provided to initialize a PolyData object:

    - a vedo mesh
    - a pyvista mesh
    - a path to a mesh
    - points, edges and triangles as torch tensors

    PolyData object has support for data attributes: point_data,
    edge_data and triangle_data. These are dictionaries of torch tensors that
    can be used to store any kind of data associated with the points, edges or
    triangles of the shape.

    Landmarks can be defined on the shape. The easiest way to define landmarks
    is to provide a list of indices.

    Landmarks can also be defined as a sparse
    tensor of shape (n_landmarks, n_points) where each line is a landmark in
    barycentric coordinates. It allows to define landmarks in a continuous
    space (on a triangle or an edge) and not only on the vertices of the shape.

    Control Points can be defined on the shape. Control Points is another
    PolyData that is used to control the deformation of the shape. It is used
    in the context of the [`ExtrinsicDeformation`](../../morphing/extrinsic_deformation)
    model.

    For visualization, PolyData can be converted into a pyvista PolyData or a
    vedo Mesh with the `to_pyvista` and `to_vedo` methods.

    Also, PolyData can be saved to a file with the `save` method. The format
    accepted by PyVista are supported (.ply, .stl, .vtk).

    Parameters
    ----------
    points
        The points of the shape. Could also be a vedo mesh, a pyvista mesh or a
        path to a file. If it is a mesh or a path, the edges and triangles are
        inferred from it.
    edges
        The edges of the shape.
    triangles
        The triangles of the shape.
    device
        The device on which the shape is stored. If None it is inferred
        from the points.
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

    See the corresponding [examples](../../../generated/gallery/#data)

    """

    @convert_inputs
    @typecheck
    def __init__(
        self,
        points: Points | vedo.Mesh | pyvista.PolyData | Path | str,
        *,
        edges: Edges | None = None,
        triangles: Triangles | None = None,
        device: str | torch.device | None = None,
        landmarks: Landmarks | IntSequence | None = None,
        control_points: polydata_type | None = None,
        stiff_edges: Edges | None = None,
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
            # Call the setter that will clone and check the validity of the
            # triangles
            self.triangles = triangles
            self._edges = None

        elif edges is not None:
            # Call the setter that will clone and check the validity of the
            # edges
            self.edges = edges
            self._triangles = None

        else:
            self._triangles = None
            self._edges = None

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

        # Cached methods: for reference on the Python syntax,
        # see "don't lru_cache methods! (intermediate) anthony explains #382",
        # https://www.youtube.com/watch?v=sVjtp6tGo0g

        self.cached_methods = [
            "point_convolution",
            "point_normals",
            "point_frames",
            "point_moments",
            "point_quadratic_coefficients",
            "point_quadratic_fits",
            "point_principal_curvatures",
            "point_shape_indices",
            "point_curvedness",
            "point_curvature_colors",
            "mesh_convolution",
        ]
        for method_name in self.cached_methods:
            setattr(
                self,
                method_name,
                functools.lru_cache(maxsize=cache_size)(
                    getattr(self, "_" + method_name)
                ),
            )

    from ..convolutions import _mesh_convolution, _point_convolution
    from ..features import (
        _point_curvature_colors,
        _point_curvedness,
        _point_frames,
        _point_moments,
        _point_normals,
        _point_principal_curvatures,
        _point_quadratic_coefficients,
        _point_quadratic_fits,
        _point_shape_indices,
    )
    from .utils import cache_clear

    @typecheck
    @one_and_only_one(["target_reduction", "n_points", "ratio"])
    def decimate(
        self,
        *,
        target_reduction: Number | None = None,
        n_points: int | None = None,
        ratio: Number | None = None,
    ) -> PolyData:
        """Decimation of the shape.

        Parameters
        ----------
        target_reduction
            The target reduction ratio.
        n_points
            The number of points to keep.

        Raises
        ------
        InputStructureError
            If both target_reduction and n_points are provided.
            If none of target_reduction and n_points are provided.

        Returns
        -------
        PolyData
            The decimated shape.
        """
        kwargs = {
            "target_reduction": target_reduction,
            "n_points": n_points,
            "ratio": ratio,
        }

        if self.is_triangle_mesh:
            from ..decimation import Decimation

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
        """Copy the shape on a given device."""
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
        """Vedo Mesh converter."""
        return vedo.Mesh(self.to_pyvista())

    ###########################
    #### PyVista interface ####
    ###########################
    @typecheck
    def to_pyvista(self) -> pyvista.PolyData:
        """Pyvista PolyData converter."""
        if self.dim == 3:
            points = self._points.detach().cpu().numpy()
        else:
            points = np.concatenate(
                [
                    self._points.detach().cpu().numpy(),
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
        """Plot the shape.

        Available backends are "pyvista" and "vedo". See the documentation of
        the corresponding plot method for the available arguments:

        - https://docs.pyvista.org/version/stable/api/core/_autosummary/pyvista.PointSet.plot.html
        - https://vedo.embl.es/docs/vedo/plotter.html#show

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
        """Edges getter.

        Returns
        -------
        Optional[Edges]
            The edges of the shape. If the shape is a triangle mesh, the edges
            are computed from the triangles. If the shape is not a triangle
            mesh, the edges are directly returned.
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
    def edges(self, edges: Edges) -> None:
        """Set the edges of the shape and set the triangles to None.

        Also reinitialize edge_data, triangle_data and the cache.
        """
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

    @typecheck
    def k_ring_graph(self, k: int = 2, verbose: bool = False) -> Edges:
        """Compute the k-ring graph of the shape.

        The k-ring graph is the graph where each vertex is connected to its
        k-neighbors, where the neighborhoods are defined by the edges.

        Parameters
        ----------
        k, optional
            the size of the neighborhood, by default 2
        verbose, optional
            whether or not display information about the remaining number of steps

        Returns
        -------
            The k-ring neighbors edges.
        """
        if not k > 0:
            msg = "The number of neighbors k must be positive"
            raise ValueError(msg)

        edges = self.edges

        if verbose:
            print(f"Computing k-neighbors graph with k = {k}")  # noqa: T201
            print(f"Step 1/{k}")  # noqa: T201

        # Compute the adjacency matrix
        adjacency_matrix = torch.zeros(
            size=(self.n_points, self.n_points),
            dtype=int_dtype,
            device=self.device,
        )
        for i in range(self.n_edges):
            i0, i1 = self.edges[i]
            adjacency_matrix[i0, i1] = 1
            adjacency_matrix[i1, i0] = 1

        M = adjacency_matrix
        k_ring_edges = edges.clone()
        k_ring_edges = torch.sort(k_ring_edges, dim=1).values

        for i in range(k - 1):

            if verbose:
                print(f"Step {i+2}/{k}")  # noqa: T201

            # Iterate the adjacency matrix
            M = M @ adjacency_matrix

            # Get the connectivity (at least one path between the two vertices)
            a, b = torch.where(M > 0)

            last_edges = torch.stack((a, b), dim=1)

            # Remove the diagonal and the duplicates with the convention
            # edges[i, 0] < edges[i, 1]
            last_edges = last_edges[
                torch.where(last_edges[:, 0] - last_edges[:, 1] < 0)
            ]

            # Concatenate the new edges and remove the duplicates
            k_ring_edges = torch.cat((k_ring_edges, last_edges), dim=0)
            k_ring_edges = torch.unique(k_ring_edges, dim=0)

        return k_ring_edges

    @typecheck
    def knn_graph(self, k: int = 2, include_edges: bool = False) -> Edges:
        """Return the k-nearest neighbors edges of a shape.

        Parameters
        ----------
        shape
            The shape to process.
        k
            The number of neighbors.
        include_edges
            If True, the edges of the shape are concatenated with the k-nearest
            neighbors edges.

        Returns
        -------
        Edges
            The k-nearest neighbors edges.
        """
        if not k > 0:
            msg = "The number of neighbors k must be positive"
            raise ValueError(msg)

        from pykeops.torch import LazyTensor

        points = self.points

        points_i = LazyTensor(points[:, None, :])  # (N, 1, 3)
        points_j = LazyTensor(points[None, :, :])  # (1, N, 3)

        D_ij = ((points_i - points_j) ** 2).sum(-1)

        out = D_ij.argKmin(K=k + 1, dim=1)

        out = out[:, 1:]

        knn_edges = torch.zeros(
            (k * self.n_points, 2), dtype=int_dtype, device=self.device
        )
        knn_edges[:, 0] = torch.repeat_interleave(
            torch.arange(self.n_points), k
        )
        knn_edges[:, 1] = out.flatten()

        if include_edges and self.edges is not None:
            knn_edges = torch.cat([self.edges, knn_edges], dim=0)

        knn_edges, _ = torch.sort(knn_edges, dim=1)

        return torch.unique(knn_edges, dim=0)

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
    def triangles(self, triangles: Triangles) -> None:
        """Set the triangles of the shape and set edges to None.

        Also reinitialize edge_data, triangle_data and the cache.
        """
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
        return self._points.shape[1]

    @property
    @typecheck
    def n_points(self) -> int:
        """Number of points getter."""
        return self._points.shape[0]

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

    @property
    @typecheck
    def mean_point(self) -> Points:
        """Mean point of the shape.

        Return the mean point as a (N_batch, 3) tensor.
        """
        # TODO: add support for batch vectors
        # TODO: add support for point weights
        return self._points.mean(dim=0, keepdim=True)

    @property
    @typecheck
    def standard_deviation(self) -> Float1dTensor:
        """Standard deviation of the shape.

        Returns the standard deviation (radius) of the shape as a (N_batch,)
        tensor.
        """
        # TODO: add support for batch vectors
        # TODO: add support for point weights
        return (
            ((self._points - self.mean_point) ** 2)
            .sum(dim=1)
            .mean(dim=0)
            .sqrt()
            .view(-1)
        )

    @property
    @typecheck
    def edge_centers(self) -> Points:
        """Center of each edge."""
        # Raise an error if edges are not defined
        if self.edges is None:
            msg = "Edges are not defined"
            raise AttributeError(msg)

        return (
            self.points[self.edges[:, 0]] + self.points[self.edges[:, 1]]
        ) / 2

    @property
    @typecheck
    def edge_lengths(self) -> Float1dTensor:
        """Length of each edge."""
        # Raise an error if edges are not defined
        if self.edges is None:
            msg = "Edges are not defined"
            raise AttributeError(msg)

        return (
            self.points[self.edges[:, 0]] - self.points[self.edges[:, 1]]
        ).norm(dim=1)

    @property
    @typecheck
    def triangle_centers(self) -> Points:
        """Center of the triangles."""
        # Raise an error if triangles are not defined
        if self.triangles is None:
            msg = "Triangles are not defined"
            raise AttributeError(msg)

        A = self.points[self.triangles[:, 0]]
        B = self.points[self.triangles[:, 1]]
        C = self.points[self.triangles[:, 2]]

        return (A + B + C) / 3

    @property
    @typecheck
    def triangle_areas(self) -> Float1dTensor:
        """Area of each triangle."""
        # Raise an error if triangles are not defined
        return self.triangle_normals.norm(dim=1) / 2

    @property
    @typecheck
    def triangle_normals(self) -> Float2dTensor:
        """Normal of each triangle."""
        # Raise an error if triangles are not defined
        if self.triangles is None:
            msg = "Triangles are not defined"
            raise AttributeError(msg)

        A = self.points[self.triangles[:, 0]]
        B = self.points[self.triangles[:, 1]]
        C = self.points[self.triangles[:, 2]]

        if self.dim == 2:
            # Add a zero z coordinate to the points to compute the
            # cross product
            A = torch.cat((A, torch.zeros_like(A[:, 0]).view(-1, 1)), dim=1)
            B = torch.cat((B, torch.zeros_like(B[:, 0]).view(-1, 1)), dim=1)
            C = torch.cat((C, torch.zeros_like(C[:, 0]).view(-1, 1)), dim=1)

        # TODO: Normalize?
        return torch.linalg.cross(B - A, C - A)

    @typecheck
    def is_triangle_mesh(self) -> bool:
        """Check if the shape is triangular."""
        return self._triangles is not None

    @property
    @typecheck
    def point_weights(self) -> Float1dTensor:
        """Point weights."""
        from ..utils import scatter

        if self.triangles is not None:
            areas = self.triangle_areas / 3
            # Triangles are stored in a (n_triangles, 3) tensor,
            # so we must repeat the areas 3 times, without interleaving.
            areas = areas.repeat(3)
            unbalanced_weights = scatter(
                index=self.triangles.flatten(),
                src=areas,
                reduce="sum",
            )

        elif self.edges is not None:
            lengths = self.edge_lengths / 2
            # Edges are stored in a (n_edges, 2) tensor,
            # so we must repeat the lengths 2 times, without interleaving.
            lengths = lengths.repeat(2)
            unbalanced_weights = scatter(
                index=self.edges.flatten(),
                src=lengths,
                reduce="sum",
            )

        else:
            unbalanced_weights = torch.ones(
                self.n_points, dtype=float_dtype, device=self.device
            )

        return unbalanced_weights

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
    def control_points(self) -> polydata_type | None:
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
        if self.triangles is not None:
            points = self.triangle_centers
            weights = self.triangle_areas / self.triangle_areas.sum()

        elif self.edges is not None:
            points = self.edge_centers
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
