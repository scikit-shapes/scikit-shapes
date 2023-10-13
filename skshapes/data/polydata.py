from __future__ import annotations

import pyvista
import vedo
import torch
import numpy as np
import functools

from ..types import (
    convert_inputs,
    typecheck,
    float_dtype,
    int_dtype,
    Number,
    NumericalArray,
    NumericalTensor,
    Points,
    Edges,
    Triangles,
    Landmarks,
    Optional,
    Union,
    Float1dTensor,
    Float2dTensor,
    Any,
    FloatTensor,
    IntTensor,
    polydata_type,
    IntSequence,
)

from typing import Literal
from .baseshape import BaseShape
from .utils import DataAttributes

from .edges_extraction import extract_edges


class PolyData(BaseShape, polydata_type):
    """A polygonal data object. It is composed of points, edges and triangles.


    Three types of objects can be provided to initialize a PolyData object:
    - a vedo mesh
    - a pyvista mesh
    - a path to a mesh
    - points, edges and triangles as torch tensors

    For all these cases, it is possible to provide landmarks as a sparse tensor and device as a string or a torch device ("cpu" by default)
    """

    @convert_inputs
    @typecheck
    def __init__(
        self,
        points: Union[Points, vedo.Mesh, pyvista.PolyData, str],
        *,
        edges: Optional[Edges] = None,
        triangles: Optional[Triangles] = None,
        device: Union[str, torch.device] = "cpu",
        landmarks: Optional[Union[Landmarks, IntSequence]] = None,
        point_data: Optional[DataAttributes] = None,
        cache_size: Optional[int] = None,
    ) -> None:
        """Initialize a PolyData object.

        Args:
            points (Points): the points of the shape.
            edges (Optional[Edges], optional): the edges of the shape. Defaults to None.
            triangles (Optional[Triangles], optional): the triangles of the shape. Defaults to None.
            device (Optional[Union[str, torch.device]], optional): the device on which the shape is stored. Defaults to "cpu".
            landmarks (Optional[Landmarks], optional): _description_. Defaults to None.
            point_data (Optional[DataAttributes], optional): _description_. Defaults to None.
            cache_size (Optional[int], optional): Size of the cache for memoized properties.
                Defaults to None (= no cache limit).
                Use a smaller value if you intend to e.g. compute point curvatures
                at many different scales.
        """

        # If the user provides a pyvista mesh, we extract the points, edges and triangles from it
        # If the user provides a path to a mesh, we read it with pyvista and extract the points, edges and triangles from it

        if type(points) in [vedo.Mesh, pyvista.PolyData, str]:
            if type(points) == vedo.Mesh:
                mesh = pyvista.PolyData(points.polydata())
            elif type(points) == str:
                mesh = pyvista.read(points)
            elif type(points) == pyvista.PolyData:
                mesh = points

            cleaned_mesh = mesh.clean()
            if cleaned_mesh.n_points != mesh.n_points:
                mesh = cleaned_mesh
                if landmarks is not None:
                    print(f"Warning: Mesh has been cleaned. Landmarks are ignored.")
                    landmarks = None

                if point_data is not None:
                    print(f"Warning: Mesh has been cleaned. Point_data are ignored.")
                    point_data = None

                if len(mesh.point_data) > 0:
                    print(
                        f"Warning: Mesh has been cleaned. Point_data from original shape are ignored."
                    )

            points = torch.from_numpy(mesh.points).to(float_dtype)

            if mesh.is_all_triangles:
                triangles = mesh.faces.reshape(-1, 4)[:, 1:]
                triangles = torch.from_numpy(triangles.copy())
                edges = None

            elif mesh.triangulate().is_all_triangles:
                triangles = mesh.triangulate().faces.reshape(-1, 4)[:, 1:]
                triangles = torch.from_numpy(triangles.copy())
                edges = None

            elif len(mesh.lines) > 0:
                edges = mesh.lines.reshape(-1, 3)[:, 1:]
                edges = torch.from_numpy(edges.copy())
                triangles = None

            else:
                edges = None
                triangles = None

            if len(mesh.point_data) > 0:
                point_data = DataAttributes.from_pyvista_datasetattributes(
                    mesh.point_data
                )

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

        if device is None:
            device = points.device

        self._device = torch.device(device)

        # We don't call the setters here because the setter of points is meant to be used when the shape is modified
        # in order to check the validity of the new points
        self._points = points.clone().to(device)

        # /!\ If triangles is not None, edges will be ignored
        if triangles is not None:
            # Call the setter that will clone and check the validity of the triangles
            self.triangles = triangles
            self._edges = None

        elif edges is not None:
            # Call the setter that will clone and check the validity of the edges
            self.edges = edges
            self._triangles = None

        else:
            self._triangles = None
            self._edges = None

        if landmarks is not None:
            # Call the setter that will clone and check the validity of the landmarks
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

        # Initialize the point_data if it was not done before
        if point_data is None:
            self._point_data = DataAttributes(n=self.n_points, device=self.device)
        else:
            assert (
                point_data.n == self.n_points
            ), "point_data must have the same number of points as the shape"
            if point_data.device != self.device:
                point_data = point_data.to(self.device)

            self._point_data = point_data

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

    from .utils import cache_clear
    from ..convolutions import _point_convolution, _mesh_convolution
    from ..features import (
        _point_normals,
        _point_frames,
        _point_moments,
        _point_quadratic_coefficients,
        _point_quadratic_fits,
        _point_principal_curvatures,
        _point_shape_indices,
        _point_curvedness,
        _point_curvature_colors,
    )

    @typecheck
    def _init_from_pyvista(self, mesh: pyvista.PolyData) -> None:
        pass

    @typecheck
    def decimate(
        self,
        *,
        target_reduction: Optional[float] = None,
        n_points: Optional[Number] = None,
    ) -> PolyData:
        """Decimate the shape using the Quadric Decimation algorithm."""
        if target_reduction is None and n_points is None:
            raise ValueError("Either target_reduction or n_points must be provided.")

        if target_reduction is not None and n_points is not None:
            raise ValueError(
                "Only one of target_reduction or n_points must be provided."
            )

        if n_points is not None:
            assert n_points > 0, "n_points must be positive"
            target_reduction = max(0, 1 - n_points / self.n_points)

        assert (
            target_reduction >= 0 and target_reduction <= 1
        ), "target_reduction must be between 0 and 1"

        from ..decimation import Decimation

        d = Decimation(target_reduction=target_reduction, n_points=n_points)
        return d.fit_transform(self)

    @typecheck
    def save(self, filename: str) -> None:
        """Save the shape at the specified location.
        Format accepted by PyVista are supported (.ply, .stl, .vtk)
        see : https://github.com/pyvista/pyvista/blob/release/0.40/pyvista/core/pointset.py#L439-L1283

        Args:
            path (str): the path where to save the shape.
        """

        mesh = self.to_pyvista()
        mesh.save(filename)

    #########################
    #### Copy functions #####
    #########################
    @typecheck
    def copy(self, device: Optional[Union[str, torch.device]] = None) -> PolyData:
        """Return a copy of the shape"""
        if device is None:
            device = self.device

        kwargs = {"points": self._points.clone(), "device": device}

        if self._triangles is not None:
            kwargs["triangles"] = self._triangles.clone()
        if self._edges is not None:
            kwargs["edges"] = self._edges.clone()
        if self._landmarks is not None:
            kwargs["landmarks"] = self._landmarks.clone()
        if self._point_data is not None:
            kwargs["point_data"] = self._point_data.clone()
        return PolyData(**kwargs)

    @typecheck
    def to(self, device: Union[str, torch.device]) -> PolyData:
        """Return a copy of the shape on the specified device"""
        return self.copy(device=device)

    ###########################
    #### Vedo interface #######
    ###########################
    @typecheck
    def to_vedo(self) -> vedo.Mesh:
        """Convert the shape to a vedo Mesh."""

        if self._triangles is not None:
            mesh = vedo.Mesh(
                [
                    self.points.detach().cpu().numpy(),
                    self.triangles.detach().cpu().numpy(),
                ]
            )
        elif self._edges is not None:
            mesh = vedo.Mesh(
                [
                    self.points.detach().cpu().numpy(),
                    self.edges.detach().cpu().numpy(),
                ]
            )
        else:
            mesh = vedo.Mesh(self.points.detach().cpu().numpy())

        # Add the point data if any
        if len(self.point_data) > 0:
            point_data_dict = self.point_data.to_numpy_dict()
            for key in point_data_dict:
                mesh.pointdata[key] = point_data_dict[key]

        # Add the landmarks if any
        if hasattr(self, "_landmarks") and self.landmarks is not None:
            coalesced_landmarks = self.landmarks.coalesce()
            mesh.metadata["landmarks_values"] = coalesced_landmarks.values()
            mesh.metadata["landmarks_indices"] = coalesced_landmarks.indices()
            mesh.metadata["landmarks_size"] = coalesced_landmarks.size()
            mesh.metadata["landmark_points"] = self.landmark_points.detach()

        return mesh

    ###########################
    #### PyVista interface ####
    ###########################
    @typecheck
    def to_pyvista(self) -> pyvista.PolyData:
        """Convert the shape to a PyVista PolyData."""

        if self._triangles is not None:
            np_triangles = self._triangles.detach().cpu().numpy()
            faces = np.concatenate(
                [np.ones((self.n_triangles, 1), dtype=np.int64) * 3, np_triangles],
                axis=1,
            )
            polydata = pyvista.PolyData(
                self._points.detach().cpu().numpy(), faces=faces
            )

        elif self._edges is not None:
            np_edges = self._edges.detach().cpu().numpy()
            lines = np.concatenate(
                [np.ones((self.n_edges, 1), dtype=np.int64) * 2, np_edges], axis=1
            )
            polydata = pyvista.PolyData(
                self._points.detach().cpu().numpy(), lines=lines
            )

        else:
            polydata = pyvista.PolyData(self._points.detach().cpu().numpy())

        # Add the point data if any
        if len(self.point_data) > 0:
            point_data_dict = self.point_data.to_numpy_dict()
            for key in point_data_dict:
                polydata.point_data[key] = point_data_dict[key]

        # Add the landmarks if any
        if hasattr(self, "_landmarks") and self.landmarks is not None:
            coalesced_landmarks = self.landmarks.coalesce()
            polydata.field_data["landmarks_values"] = coalesced_landmarks.values()
            polydata.field_data["landmarks_indices"] = coalesced_landmarks.indices()
            polydata.field_data["landmarks_size"] = coalesced_landmarks.size()
            polydata.field_data["landmark_points"] = self.landmark_points.detach()

        return polydata

    @classmethod
    @typecheck
    def from_pyvista(
        cls, mesh: pyvista.PolyData, device: Optional[Union[str, torch.device]] = "cpu"
    ) -> PolyData:
        import warnings

        warnings.warn(
            "from_pyvista is deprecated, use PolyData(mesh) instead", DeprecationWarning
        )
        """Create a Shape from a PyVista PolyData object."""

        points = torch.from_numpy(mesh.points).to(float_dtype)

        if mesh.is_all_triangles:
            triangles = mesh.faces.reshape(-1, 4)[:, 1:]
            triangles = torch.from_numpy(triangles.copy())
            edges = None

        elif mesh.triangulate().is_all_triangles:
            triangles = mesh.triangulate().faces.reshape(-1, 4)[:, 1:]
            triangles = torch.from_numpy(triangles.copy())
            edges = None

        elif len(mesh.lines) > 0:
            edges = mesh.lines.reshape(-1, 3)[:, 1:]
            edges = torch.from_numpy(edges.copy())
            triangles = None

        else:
            edges = None
            triangles = None

        return cls(points=points, edges=edges, triangles=triangles, device=device)

    #############################
    #### Edges getter/setter ####
    #############################
    @property
    @typecheck
    def edges(self) -> Optional[Edges]:
        if self._edges is not None:
            return self._edges

        elif self._triangles is not None:
            points_numpy = self.points.detach().cpu().numpy().astype(np.float64)
            triangles_numpy = self.triangles.detach().cpu().numpy().astype(np.int64)
            edges = extract_edges(points_numpy, triangles_numpy.T)
            edges = torch.from_numpy(edges.T).to(int_dtype).to(self.device)

            self._edges = edges
            return edges

    @edges.setter
    @convert_inputs
    @typecheck
    def edges(self, edges: Edges) -> None:
        """Set the edges of the shape. This will also set the triangles to None."""
        if edges.max() >= self.n_points:
            raise ValueError(
                "The maximum vertex index in the triangles is larger than the number of points."
            )
        self._edges = edges.clone().to(self.device)
        self._triangles = None
        self.cache_clear()

    #################################
    #### Triangles getter/setter ####
    #################################
    @property
    @typecheck
    def triangles(self) -> Optional[Triangles]:
        return self._triangles

    @triangles.setter
    @convert_inputs
    @typecheck
    def triangles(self, triangles: Triangles) -> None:
        """Set the triangles of the shape. This will also set the edges to None."""
        if triangles.max() >= self.n_points:
            raise ValueError(
                "The maximum vertex index in the triangles is larger than the number of points."
            )

        self._triangles = triangles.clone().to(self.device)
        self._edges = None
        self.cache_clear()

    ##############################
    #### Points getter/setter ####
    ##############################
    @property
    @typecheck
    def points(self) -> Points:
        return self._points

    @points.setter
    @convert_inputs
    @typecheck
    def points(self, points: Points) -> None:
        if points.shape[0] != self.n_points:
            raise ValueError("The number of points cannot be changed.")

        self._points = points.clone().to(self.device)
        self.cache_clear()

    ##############################
    #### Device getter/setter ####
    ##############################
    @property
    @typecheck
    def device(self) -> Union[str, torch.device]:
        return self._device

    @device.setter
    @typecheck
    def device(self, device: Union[str, torch.device]) -> None:
        for attr in dir(self):
            if attr.startswith("_"):
                if type(getattr(self, attr)) == torch.Tensor:
                    setattr(self, attr, getattr(self, attr).to(device))
        self._device = device

    ################################
    #### point_data getter/setter ##
    ################################
    @property
    @typecheck
    def point_data(self) -> DataAttributes:
        return self._point_data

    @point_data.setter
    @typecheck
    def point_data(self, point_data_dict: dict) -> None:
        if not isinstance(point_data_dict, DataAttributes):
            # Convert the point_data to a DataAttributes object
            # the from_dict method will check that the point_data are valid
            point_data_dict = DataAttributes.from_dict(point_data_dict)

        assert (
            point_data_dict.n == self.n_points
        ), "The number of points in the point_data entries should be the same as the number of points in the shape."
        self._point_data = point_data_dict.to(self.device)

    @typecheck
    def __getitem__(self, key: Any) -> NumericalTensor:
        """Return the point data corresponding to the key."""
        if key not in self._point_data:
            raise KeyError(f"Point data {key} is not defined.")
        return self._point_data[key]

    @convert_inputs
    @typecheck
    def __setitem__(self, key: Any, value: NumericalTensor) -> None:
        """Set the point data corresponding to the key."""
        self._point_data[key] = value

    #################################
    #### Landmarks getter/setter ####
    #################################
    @property
    @typecheck
    def landmarks(self) -> Optional[Landmarks]:
        """Get the landmarks of the shape.

        The format is a sparse tensor of shape (n_landmarks, n_points),each line is a landmark in barycentric
        coordinates. If you want to get the landmarks in 3D coordinates, use the landmark_points property. If
        you want to get the landmarks as a list of indices, use the landmark_indices property.

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
    @typecheck
    def landmarks(self, landmarks: Landmarks) -> None:
        """Set the landmarks of the shape. The landmarks should be a sparse tensor of shape
        (n_landmarks, n_points) (barycentric coordinates) or a list of indices."""

        assert landmarks.is_sparse and landmarks.shape[1] == self.n_points
        assert landmarks.dtype == float_dtype
        self._landmarks = landmarks.clone().to(self.device)

    @property
    @typecheck
    def landmark_points(self) -> Optional[Points]:
        """Return the landmarks in 3D coordinates."""
        if self.landmarks is None:
            return None
        else:
            return self.landmarks @ self.points

    @property
    @typecheck
    def landmark_indices(self) -> Optional[IntTensor]:
        """Return the indices of the landmarks.

        If no landmarks are defined, returns None.
        Raises an error if the landmarks are not indices (there are defined in barycentric coordinates).
        """
        if self.landmarks is None:
            return None
        else:
            coalesced_landmarks = self.landmarks.coalesce()
            values = coalesced_landmarks.values()
            indices = coalesced_landmarks.indices()[1][values == 1]

            if len(indices) != self.n_landmarks:
                raise ValueError("Landmarks are not indices.")

            return indices[values == 1]

    @landmark_indices.setter
    @typecheck
    def landmark_indices(self, landmarks: IntSequence) -> None:
        """Set the landmarks of the shape. The landmarks should be a list of indices."""

        assert torch.max(torch.tensor(landmarks)) < self.n_points

        n_landmarks = len(landmarks)
        n_points = self.n_points

        indices = torch.zeros((2, n_landmarks), dtype=int_dtype)
        indices[0] = torch.arange(n_landmarks, dtype=int_dtype)
        indices[1] = torch.tensor(landmarks, dtype=int_dtype)

        values = torch.ones_like(indices[0], dtype=float_dtype)

        self.landmarks = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(n_landmarks, n_points),
            device=self.device,
        )

    def add_landmarks(self, indices: Union[IntSequence, int]) -> None:
        """Add vertices landmarks to the shape.

        Args:
            indices (Union[IntSequence, int]): the indices of the vertices to landmark.
        """
        if not hasattr(indices, "__iter__"):
            self.add_landmarks([indices])

        else:
            if self.landmarks is None:
                self.landmarks = indices

            else:
                new_indices = torch.tensor(indices, dtype=int_dtype, device=self.device)

                coalesced_landmarks = self.landmarks.coalesce()
                old_values = coalesced_landmarks.values()
                old_indices = coalesced_landmarks.indices()

                n_new_landmarks = len(new_indices)
                new_indices = torch.zeros(
                    (2, n_new_landmarks), dtype=int_dtype, device=self.device
                )
                new_indices[0] = (
                    torch.arange(n_new_landmarks, dtype=int_dtype) + self.n_landmarks
                )
                new_indices[1] = torch.tensor(
                    indices, dtype=int_dtype, device=self.device
                )

                new_values = torch.ones_like(
                    new_indices[0], dtype=float_dtype, device=self.device
                )

                n_landmarks = self.n_landmarks + n_new_landmarks
                n_points = self.n_points

                indices = torch.cat((old_indices, new_indices), dim=1)
                values = torch.concat((old_values, new_values))

                print(indices)
                print(values)
                print((n_landmarks, n_points))

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
    def n_points(self) -> int:
        return self._points.shape[0]

    @property
    @typecheck
    def n_edges(self) -> int:
        edges = self.edges
        if edges is not None:
            return edges.shape[0]
        else:
            return 0

    @property
    @typecheck
    def n_triangles(self) -> int:
        if self._triangles is not None:
            return self._triangles.shape[0]
        else:
            return 0

    @property
    @typecheck
    def mean_point(self) -> Points:
        """Returns the mean point of the shape as a (N_batch, 3) tensor."""
        # TODO: add support for batch vectors
        # TODO: add support for point weights
        return self._points.mean(dim=0, keepdim=True)

    @property
    @typecheck
    def standard_deviation(self) -> Float1dTensor:
        """Returns the standard deviation (radius) of the shape as a (N_batch,) tensor."""
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
        """Return the center of each edge"""

        # Raise an error if edges are not defined
        if self.edges is None:
            raise ValueError("Edges cannot be computed")

        return (self.points[self.edges[:, 0]] + self.points[self.edges[:, 1]]) / 2

    @property
    @typecheck
    def edge_lengths(self) -> Float1dTensor:
        """Return the length of each edge"""

        # Raise an error if edges are not defined
        if self.edges is None:
            raise ValueError("Edges cannot be computed")

        return (self.points[self.edges[:, 0]] - self.points[self.edges[:, 1]]).norm(
            dim=1
        )

    @property
    @typecheck
    def triangle_centers(self) -> Points:
        """Return the center of the triangles"""

        # Raise an error if triangles are not defined
        if self.triangles is None:
            raise ValueError("Triangles are not defined")

        A = self.points[self.triangles[:, 0]]
        B = self.points[self.triangles[:, 1]]
        C = self.points[self.triangles[:, 2]]

        return (A + B + C) / 3

    @property
    @typecheck
    def triangle_areas(self) -> Float1dTensor:
        """Return the area of each triangle"""

        # Raise an error if triangles are not defined
        if self.triangles is None:
            raise ValueError("Triangles are not defined")

        A = self.points[self.triangles[:, 0]]
        B = self.points[self.triangles[:, 1]]
        C = self.points[self.triangles[:, 2]]

        return torch.cross(B - A, C - A).norm(dim=1) / 2

    @property
    @typecheck
    def triangle_normals(self) -> Float2dTensor:
        """Return the normal of each triangle"""

        # Raise an error if triangles are not defined
        if self.triangles is None:
            raise ValueError("Triangles are not defined")

        A = self.points[self.triangles[:, 0]]
        B = self.points[self.triangles[:, 1]]
        C = self.points[self.triangles[:, 2]]

        # TODO: Normalize?
        return torch.cross(B - A, C - A)

    @typecheck
    def is_triangle_mesh(self) -> bool:
        return self._triangles is not None

    @property
    @typecheck
    def point_weights(self) -> Float1dTensor:
        """Return the weights of each point"""
        if self.triangles is not None:
            areas = self.triangle_areas / 3
            # Triangles are stored in a (3, n_triangles) tensor,
            # so we must repeat the areas 3 times, without interleaving.
            areas = areas.repeat(3)
            return torch.bincount(
                self.triangles.T.flatten(), weights=areas, minlength=self.n_points
            )

        elif self.edges is not None:
            lengths = self.edge_lengths / 2
            # Edges are stored in a (2, n_edges) tensor,
            # so we must repeat the lengths 2 times, without interleaving.
            lengths = lengths.repeat(2)
            return torch.bincount(
                self.edges.T.flatten(), weights=lengths, minlength=self.n_points
            )

        return torch.ones(self.n_points, dtype=float_dtype, device=self.device)
