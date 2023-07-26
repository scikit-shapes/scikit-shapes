from __future__ import annotations

import pyvista
import vedo
import torch
import numpy as np

from ..types import (
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
)


from .baseshape import BaseShape
from .utils import DataAttributes


class PolyData(BaseShape):
    """A polygonal data object. It is composed of points, edges and triangles.


    Three types of objects can be provided to initialize a PolyData object:
    - a vedo mesh
    - a pyvista mesh
    - a path to a mesh
    - points, edges and triangles as torch tensors

    For all these cases, it is possible to provide landmarks as a sparse tensor and device as a string or a torch device ("cpu" by default)
    """

    @typecheck
    def __init__(
        self,
        points: Union[Points, vedo.Mesh, pyvista.PolyData, str],
        *,
        edges: Optional[Edges] = None,
        triangles: Optional[Triangles] = None,
        device: Union[str, torch.device] = "cpu",
        landmarks: Optional[Landmarks] = None,
        point_data: Optional[DataAttributes] = None,
    ) -> None:
        """Initialize a PolyData object.

        Args:
            points (Points): the points of the shape.
            edges (Optional[Edges], optional): the edges of the shape. Defaults to None.
            triangles (Optional[Triangles], optional): the triangles of the shape. Defaults to None.
            device (Optional[Union[str, torch.device]], optional): the device on which the shape is stored. Defaults to "cpu".
            landmarks (Optional[Landmarks], optional): _description_. Defaults to None.
            point_data (Optional[DataAttributes], optional): _description_. Defaults to None.
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
                    point_data = None

                if point_data is not None:
                    print(f"Warning: Mesh has been cleaned. Point_data are ignored.")
                    point_data = None

                if len(mesh.point_data) > 0:
                    print(
                        f"Warning: Mesh has been cleaned. Point_data from original shape are ignored."
                    )

            points = torch.from_numpy(mesh.points).to(float_dtype)

            if mesh.is_all_triangles:
                triangles = mesh.faces.reshape(-1, 4)[:, 1:].T
                triangles = torch.from_numpy(triangles.copy())
                edges = None

            elif mesh.triangulate().is_all_triangles:
                triangles = mesh.triangulate().faces.reshape(-1, 4)[:, 1:].T
                triangles = torch.from_numpy(triangles.copy())
                edges = None

            elif len(mesh.lines) > 0:
                edges = mesh.lines.reshape(-1, 3)[:, 1:].T
                edges = torch.from_numpy(edges.copy())
                triangles = None

            else:
                edges = None
                triangles = None

            if len(mesh.point_data) > 0:
                point_data = DataAttributes.from_pyvista_datasetattributes(
                    mesh.point_data
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

        elif edges is not None:
            # Call the setter that will clone and check the validity of the edges
            self.edges = edges

        else:
            self._triangles = None
            self._edges = None

        if landmarks is not None:
            # Call the setter that will clone and check the validity of the landmarks
            self.landmarks = landmarks

        else:
            self._landmarks = None

        # Initialize the point_data
        if point_data is None:
            self._point_data = DataAttributes(n=self.n_points, device=self.device)
        else:
            assert (
                point_data.n == self.n_points
            ), "point_data must have the same number of points as the shape"
            if point_data.device != self.device:
                point_data = point_data.to(self.device)

            self._point_data = point_data

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
        """Decimate the shape using the Quadric Decimation algorithm.

        Args:
            target_reduction (float): the target reduction ratio. Must be between 0 and 1.

        Returns:
            PolyData: the decimated PolyData
        """
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

        mesh = self.to_pyvista()
        mesh = mesh.decimate(target_reduction)
        return PolyData(mesh, device=self.device)

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
                    self.triangles.detach().cpu().numpy().T,
                ]
            )
        elif self._edges is not None:
            mesh = vedo.Mesh(
                [
                    self.points.detach().cpu().numpy(),
                    self.edges.detach().cpu().numpy().T,
                ]
            )
        else:
            mesh = vedo.Mesh(self.points.detach().cpu().numpy())

        # Add the point data if any
        if len(self.point_data) > 0:
            point_data_dict = self.point_data.to_numpy_dict()
            for key in point_data_dict:
                mesh.pointdata[key] = point_data_dict[key]

        return mesh

    ###########################
    #### PyVista interface ####
    ###########################
    @typecheck
    def to_pyvista(self) -> pyvista.PolyData:
        """Convert the shape to a PyVista PolyData."""

        if self._triangles is not None:
            np_triangles = self._triangles.detach().cpu().numpy().T
            faces = np.concatenate(
                [np.ones((np_triangles.shape[0], 1), dtype=np.int64) * 3, np_triangles],
                axis=1,
            )
            polydata = pyvista.PolyData(
                self._points.detach().cpu().numpy(), faces=faces
            )

        elif self._edges is not None:
            np_edges = self._edges.detach().cpu().numpy().T
            lines = np.concatenate(
                [np.ones((np_edges.shape[0], 1), dtype=np.int64) * 2, np_edges], axis=1
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
            triangles = mesh.faces.reshape(-1, 4)[:, 1:].T
            triangles = torch.from_numpy(triangles.copy())
            edges = None

        elif mesh.triangulate().is_all_triangles:
            triangles = mesh.triangulate().faces.reshape(-1, 4)[:, 1:].T
            triangles = torch.from_numpy(triangles.copy())
            edges = None

        elif len(mesh.lines) > 0:
            edges = mesh.lines.reshape(-1, 3)[:, 1:].T
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
            # Compute the edges of the triangles and sort them
            repeated_edges = torch.concat(
                [
                    self.triangles[[0, 1], :],
                    self.triangles[[1, 2], :],
                    self.triangles[[0, 2], :],
                ],
                dim=1,
            ).sort(dim=0)[0]

            # Remove the duplicates and return
            return torch.unique(repeated_edges, dim=1)

    @edges.setter
    @typecheck
    def edges(self, edges: Edges) -> None:
        """Set the edges of the shape. This will also set the triangles to None."""
        self._edges = edges.clone().to(self.device)
        self._triangles = None

    #################################
    #### Triangles getter/setter ####
    #################################
    @property
    @typecheck
    def triangles(self) -> Optional[Triangles]:
        return self._triangles

    @triangles.setter
    @typecheck
    def triangles(self, triangles: Triangles) -> None:
        """Set the triangles of the shape. This will also set the edges to None."""
        if triangles.max() >= self.n_points:
            raise ValueError(
                "The maximum vertex index in the triangles is larger than the number of points."
            )

        self._triangles = triangles.clone().to(self.device)
        self._edges = None

    ##############################
    #### Points getter/setter ####
    ##############################
    @property
    @typecheck
    def points(self) -> Points:
        print("get points")
        return self._points

    @points.setter
    @typecheck
    def points(self, points: Points) -> None:
        print("set points")
        if points.shape[0] != self.n_points:
            raise ValueError("The number of points cannot be changed.")

        self._points = points.clone().to(self.device)

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

    #################################
    #### Landmarks getter/setter ####
    #################################
    @property
    @typecheck
    def landmarks(self) -> Optional[Landmarks]:
        return self._landmarks

    @landmarks.setter
    @typecheck
    def landmarks(self, landmarks: Landmarks) -> None:
        """Set the landmarks of the shape. The landmarks should be a sparse tensor of shape
        (n_landmarks, n_points) (barycentric coordinates) or a ."""
        assert landmarks.is_sparse and landmarks.shape[1] == self.n_points
        assert landmarks.dtype == float_dtype

        self._landmarks = landmarks.clone().to(self.device)

    @property
    @typecheck
    def landmarks_3d(self) -> Optional[Points]:
        """Return the landmarks in 3D."""
        if self.landmarks is None:
            return None
        else:
            return self.landmarks @ self.points

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
            return edges.shape[1]
        else:
            return 0

    @property
    @typecheck
    def n_triangles(self) -> int:
        if self._triangles is not None:
            return self._triangles.shape[1]
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

        return (self.points[self.edges[0]] + self.points[self.edges[1]]) / 2

    @property
    @typecheck
    def edge_lengths(self) -> Float1dTensor:
        """Return the length of each edge"""

        # Raise an error if edges are not defined
        if self.edges is None:
            raise ValueError("Edges cannot be computed")

        return (self.points[self.edges[0]] - self.points[self.edges[1]]).norm(dim=1)

    @property
    @typecheck
    def triangle_centers(self) -> Points:
        """Return the center of the triangles"""

        # Raise an error if triangles are not defined
        if self.triangles is None:
            raise ValueError("Triangles are not defined")

        A = self.points[self.triangles[0]]
        B = self.points[self.triangles[1]]
        C = self.points[self.triangles[2]]

        return (A + B + C) / 3

    @property
    @typecheck
    def triangle_areas(self) -> Float1dTensor:
        """Return the area of each triangle"""

        # Raise an error if triangles are not defined
        if self.triangles is None:
            raise ValueError("Triangles are not defined")

        A = self.points[self.triangles[0]]
        B = self.points[self.triangles[1]]
        C = self.points[self.triangles[2]]

        return torch.cross(B - A, C - A).norm(dim=1) / 2

    @property
    @typecheck
    def triangle_normals(self) -> Float2dTensor:
        """Return the normal of each triangle"""

        # Raise an error if triangles are not defined
        if self.triangles is None:
            raise ValueError("Triangles are not defined")

        A = self.points[self.triangles[0]]
        B = self.points[self.triangles[1]]
        C = self.points[self.triangles[2]]

        return torch.cross(B - A, C - A)

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
                self.triangles.flatten(), weights=areas, minlength=self.n_points
            )

        elif self.edges is not None:
            lengths = self.edge_lengths / 2
            # Edges are stored in a (2, n_edges) tensor,
            # so we must repeat the lengths 2 times, without interleaving.
            lengths = lengths.repeat(2)
            return torch.bincount(
                self.edges.flatten(), weights=lengths, minlength=self.n_points
            )

        return torch.ones(self.n_points, dtype=float_dtype, device=self.device)

    from ..convolutions import point_convolution
    from ..features import point_moments
