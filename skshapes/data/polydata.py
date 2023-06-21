import pyvista
import torch
import numpy as np

from ..types import (
    typecheck,
    PolyData,
    float_dtype,
    Points,
    Edges,
    Triangles,
    Landmarks,
    Optional,
    Union,
    Float1dTensor,
    Float2dTensor,
)


@typecheck
def read(filename: str) -> PolyData:

    mesh = pyvista.read(filename)
    if type(mesh) == pyvista.PolyData:
        return PolyData.from_pyvista(mesh)
    else:
        raise NotImplementedError("Images are not supported yet")


class PolyData(PolyData):
    """A class to represent a surface mesh as a set of points, edges and/or triangles."""

    @typecheck
    def __init__(
        self,
        points: Points,
        edges: Optional[Edges] = None,
        triangles: Optional[Triangles] = None,
        device: Optional[Union[str, torch.device]] = None,
        landmarks: Optional[Landmarks] = None,
    ) -> None:
        """Initialize a PolyData object.

        Args:
            points (Points): the points of the shape.
            edges (Optional[Edges], optional): the edges of the shape. Defaults to None.
            triangles (Optional[Triangles], optional): the triangles of the shape. Defaults to None.
            device (Optional[Union[str, torch.device]], optional): the device on which the shape is stored. Defaults to None.
            landmarks (Optional[Landmarks], optional): _description_. Defaults to None.
        """

        if device is None:
            device = points.device

        self._device = device

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
        return PolyData(**kwargs)

    @typecheck
    def to(self, device: Union[str, torch.device]) -> PolyData:
        """Return a copy of the shape on the specified device"""
        return self.copy(device=device)

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
            return pyvista.PolyData(self._points.detach().cpu().numpy(), faces=faces)

        elif self._edges is not None:
            np_edges = self._edges.detach().cpu().numpy().T
            lines = np.concatenate(
                [np.ones((np_edges.shape[0], 1), dtype=np.int64) * 2, np_edges], axis=1
            )
            return pyvista.PolyData(self._points.detach().cpu().numpy(), lines=lines)

        else:
            return pyvista.PolyData(self._points.detach().cpu().numpy())

    @classmethod
    @typecheck
    def from_pyvista(
        cls, mesh: pyvista.PolyData, device: Optional[Union[str, torch.device]] = None
    ) -> PolyData:
        """Create a Shape from a PyVista PolyData object."""

        points = torch.from_numpy(mesh.points)

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
        return self._points

    @points.setter
    @typecheck
    def points(self, points: Points) -> None:
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
    def landmarks_3d(self) -> Optional[Landmarks]:
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