import torch
import pyvista
import numpy as np
import numba

def read(filename, affine=None, device='cpu'):

    #First, read the file with pyvista
    pyvista_object = pyvista.read(filename)

    # If the object is a PolyData, we extract the faces
    if isinstance(pyvista_object, pyvista.PolyData):
        faces = torch.from_numpy(pyvista_object.faces)
        points = torch.from_numpy(pyvista_object.points)
        return Shape(points=points, faces=faces, device=device)

    # If the object is a structured grid, it's an image
    elif isinstance(pyvista_object, pyvista.StructuredGrid):
        #Not implemented yet
        raise NotImplementedError

@numba.jit
def _read_faces(faces):
    len_faces = len(faces)
    i  = 0
    edges = []
    triangles = []
    while i < len_faces:
        #Get the number of points in the face
        n = faces[i]
        if n == 2:
            #Get the points of the edge
            face = faces[i+1:i+n+1]
            edges.append(face)
        elif n == 3:
            #Get the points of the triangle
            face = faces[i+1:i+n+1]
            triangles.append(face)

        i += n + 1
    
    return edges, triangles


@numba.jit(nopython=True, fastmath=True)
def _sort_edges(edges, inverse_ordering):
    """Sort the edges of a mesh, given inverse ordering
    Args:
        edges (array): the edges of the mesh
        inverse_ordering (list): the inverse order
    Returns:
        sorted_edges (array): the sorted edges
    """
    sorted_edges = np.repeat(2, len(edges))
    for i in range(len(edges)):
        if i % 3 != 0:
            sorted_edges[i] = inverse_ordering[edges[i]]

    return sorted_edges

def _edges_from_triangles(triangles):
    """Get the edges of a shape from its triangles
    Args:
        triangles (3, K): the triangles of the shape
    Returns:
        edges (2, M): the edges of the shape"""

    # - 1 construct a pyvista object from the triangles in order to extract the edges

    # Convert the triangles to a faces list 
    faces = torch.zeros((triangles.shape[1], 4), dtype=torch.int64)
    faces[:, 0] = 3
    faces[:, 1:] = triangles.T
    faces = faces.reshape(-1)

    # Generate arbitrary points in order to create a pyvista object
    points = torch.rand((triangles.max() + 1, 3))

    # Keep the lexicographic ordering of the points
    points_ordering = np.lexsort(
        (
            points[:, 2].cpu().numpy(),
            points[:, 1].cpu().numpy(),
            points[:, 0].cpu().numpy(),
        )
    )

    # - 2 Extract the edges using pyvista (does not preserve the points labelling)
    edges_mesh = pyvista.PolyData(points.numpy(), faces=faces.numpy()).extract_all_edges()

    # - 3 Sort the edges in order to retrieve the original labelling
    edges_ordering = np.lexsort(
        (
            edges_mesh.points[:, 2],
            edges_mesh.points[:, 1],
            edges_mesh.points[:, 0],
        )
    )
    inverse_edges_ordering = np.argsort(edges_ordering)

    edges = _sort_edges(
        edges_mesh.lines, inverse_edges_ordering
    )  # To lexicographic order
    edges = _sort_edges(edges, points_ordering)  # Back to the original order
    edges = edges.reshape(-1, 3)[:, 1:]  # Remove padding
    edges = torch.Tensor(edges).T.long()

    return edges


def _edges_and_triangles_from_faces(faces):

    # We convert the faces to a list (works wether for torch tensor or numpy array)
    # then we convert it to a numba typed list to avoid warning in numba
    edges, triangles = _read_faces(faces.cpu().numpy())

    # return edges, triangles
    if len(edges) == 0:
        edges = None
    else:
        edges = torch.from_numpy(np.array(edges)).T
    
    if len(triangles) == 0:
        triangles = None
    else:
        triangles = torch.from_numpy(np.array(triangles)).T

    return edges, triangles



class Shape:
    """A class for representing a shape. Shapes can have those attributes:
    - points : (N, 3) or (N, 2) array of points
    - faces (F, ) int array that can store arbitrary cell structure with padding
    - edges (optionnal, default to None) : (2, M) array of edges
    - triangles (optionnal, default to None) : (3, K) array of faces
    """
    
    def __init__(
        self,
        *,
        points=None,
        faces=None,
        edges=None,
        triangles=None,
        device="cpu",
        **kwargs,
        ) -> None:

        self.points = points
        self.faces = faces
        self.edges = edges
        self.triangles = triangles
        self.device = device

        # If edges and triangles are not defined, we compute them from the faces
        if self.edges is None and self.triangles is None and self.faces is not None:

            self.edges, self.triangles = _edges_and_triangles_from_faces(faces)
        
        # If edges are not defined, we compute them from the triangles
        # TODO : do we want to impose that edges are defined if triangles are defined ?
        if self.edges is None and self.triangles is not None:
            self.edges = _edges_from_triangles(self.triangles)

        # Pass tensors to the right device
        self.to(device)

    def copy(self):
        """Return a copy of the shape"""
        return Shape(
            points=self.points.clone(),
            faces=self.faces.clone(),
            edges=self.edges.clone(),
            triangles=self.triangles.clone(),
            device=self.device,
        )

    def to(self, device):
        """Move the shape to the specified device"""
        if self.points is not None:
            self.points = self.points.to(device)
        if self.faces is not None:
            self.faces = self.faces.to(device)
        if self.edges is not None:
            self.edges = self.edges.to(device)
        if self.triangles is not None:
            self.triangles = self.triangles.to(device)
        self.device = device

        return self
    
    def to_pyvista(self):
        """Return a pyvista object from the shape"""
        return pyvista.PolyData(self.points.cpu().numpy(), faces=self.faces.cpu().numpy())
    
    @classmethod
    def from_pyvista(cls, pyvista_object):
        """Construct a shape from a pyvista mesh"""
        return cls(
            points=torch.from_numpy(pyvista_object.points),
            faces=torch.from_numpy(pyvista_object.faces),
        )


    @property
    def edge_centers(self):
        """Return the center of each edge"""

        #Raise an error if edges are not defined
        if self.edges is None:
            raise ValueError("Edges are not defined")

        return (self.points[self.edges[0]] + self.points[self.edges[1]]) / 2
    
    @property
    def edge_lengths(self):
        """Return the length of each edge"""

        #Raise an error if edges are not defined
        if self.edges is None:
            raise ValueError("Edges are not defined")

        return (self.points[self.edges[0]] - self.points[self.edges[1]]).norm(dim=1)

    @property
    def triangle_centers(self):
        """Return the center of each triangle"""

        #Raise an error if triangles are not defined
        if self.triangles is None:
            raise ValueError("Triangles are not defined")

        A = self.points[self.triangles[0]]
        B = self.points[self.triangles[1]]
        C = self.points[self.triangles[2]]

        return (A + B + C) / 3

    @property
    def triangle_areas(self):
        """Return the area of each triangle"""

        #Raise an error if triangles are not defined
        if self.triangles is None:
            raise ValueError("Triangles are not defined")

        A = self.points[self.triangles[0]]
        B = self.points[self.triangles[1]]
        C = self.points[self.triangles[2]]

        return torch.cross(B - A, C - A).norm(dim=1) / 2