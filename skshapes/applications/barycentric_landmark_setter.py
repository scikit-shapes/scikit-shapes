import vedo
import numpy as np
import torch

from ..types import typecheck, List, float_dtype, int_dtype
from ..data import PolyData


class BarycentricLandmarkSetter(vedo.Plotter):
    """!! This class has bugs and is not used in the library. !!
    A LandmarkSetter is a vedo application that allows the user to select landmarks on a set of meshes.

    This version allows to select landmarks that are barycentric coordinates of the vertices of the mesh.

    Args:
        meshes (List[vedo.Mesh]): The meshes on which the landmarks are selected.
        **kwargs: Keyword arguments passed to the vedo.Plotter constructor.
    """

    @typecheck
    def __init__(self, meshes: List[PolyData], **kwargs) -> None:
        super().__init__(N=2, sharecam=False, **kwargs)

        # The 3D landmarks are stored in a list of lists of 3D points
        self.landmarks3d = [[] for i in range(len(meshes))]

        self.original_meshes = meshes

        # Convert the meshes to vedo.Mesh objects
        self.meshes = meshes
        self.actors = [vedo.Mesh(mesh.to_pyvista()) for mesh in meshes]

        # The first mesh is the reference
        self.reference = self.actors[0]
        self.others = self.actors[1:]

        # At the beginning : the reference mesh is plotted on the left
        # and the first other mesh is plotted on the right
        self.current_other = self.others[0]
        self.at(0).add(self.reference.linewidth(1))
        self.at(1).add(self.current_other.linewidth(1))

        # At the beginning, we are in 'reference' mode, meaning that we are
        # selecting landmarks on the reference mesh
        self.active = 0
        self.active_actor = self.reference
        self.reference_lpoints = []
        # The reference landmarks are stored in a vedo.Points object
        # for display purposes
        self.reference_lpoints_pointcloud = (
            vedo.Points(self.reference_lpoints, r=15).pickable(False).c("r")
        )
        self.mode = "reference"
        # The reference vertices are stored in a vedo.Points object, we do not display them but we store them to be able to
        # pick them
        self.reference_vertices = vedo.Points(self.reference.points())

        # Instructions corresponding to the "reference" mode
        text_reference = "Start by selecting landmarks on the reference mesh\nPress z to add a point on the surface\nPress e to add a vertice\nPress d to delete the last point\nPress s when you are done"
        self.instructions_reference = vedo.Text2D(
            text_reference, pos="bottom-left", c="white", bg="green", font="Calco"
        )
        self.at(0).add(
            self.instructions_reference
        )  # Add the instructions to the left plot

        # Instructions corresponding to the "other" mode (not displayed at the beginning)
        text_other = "Now select the same landmarks on the other meshes\nPress z to add a point on the surface\nPress e to add a vertice\nPress d to delete the last point\nPress s when you are done"
        self.instructions_other = vedo.Text2D(
            text_other, pos="bottom-left", c="white", bg="green", font="Calco"
        )

        # Add the callback keypress
        self.add_callback("KeyPress", self._key_press)

    def start(self):
        """Start the landmark setter."""
        self._update()
        self.at(0).reset_camera()
        self.at(1).reset_camera()
        self.show(interactive=True)
        # return self

    def _done(self):
        """The _done method is called when the user presses the 's' key.
        If the current mode is 'reference', it stores information about the number of landmarks to be set on the other meshes and switches to 'others' mode.
        If the current mode is 'others', it stores the landmarks for the current mesh and switches to the next mesh. If no other mesh is left, it closes the window.
        """

        if self.mode == "reference":
            self.n_landmarks = len(self.reference_lpoints)
            self.mode = "others"
            self.at(0).remove(self.instructions_reference)
            self.at(1).add(self.instructions_other)
            self.other_id = 0
            self.other_lpoints = []

            self.current_other = self.others[self.other_id]

            self.active_actor = self.current_other
            self.other_lpoints_pointcloud = (
                vedo.Points(self.other_lpoints, r=15).pickable(False).c("r")
            )
            self.reference_lpoints_pointcloud.c("grey")
            self.point_to_pick = (
                vedo.Points([self.reference_lpoints[len(self.other_lpoints)]], r=15)
                .pickable(False)
                .c("green")
            )

            self.landmarks3d[0] = self.reference_lpoints
            self._update()

        else:
            self.landmarks3d[self.other_id + 1] = self.other_lpoints.copy()
            self.other_lpoints = []
            self.other_id += 1

            if self.other_id < len(self.others):
                self.current_other = self.others[self.other_id]
                self.active_actor = self.current_other

                self.at(1).clear()
                self.at(1).add(self.current_other.linewidth(1))

                self.other_lpoints_pointcloud = (
                    vedo.Points(self.other_lpoints, r=15).pickable(False).c("r")
                )
                self._update()

            else:
                self.close()

                ls = self.landmarks
                for i in range(len(ls)):
                    self.original_meshes[i].landmarks = ls[i]

    def _update(self):
        """The _update method update the display of the landmarks with the right color depending on the current mode and the current state of the landmarks selection."""

        if self.mode == "reference":
            self.at(0).remove(self.reference_lpoints_pointcloud)
            self.reference_lpoints_pointcloud = (
                vedo.Points(self.reference_lpoints, r=15).pickable(False).c("r")
            )
            self.at(0).add(self.reference_lpoints_pointcloud)

        else:
            self.at(0).remove(self.point_to_pick)
            if len(self.other_lpoints) < self.n_landmarks:
                self.point_to_pick = (
                    vedo.Points([self.reference_lpoints[len(self.other_lpoints)]], r=15)
                    .pickable(False)
                    .c("green")
                )
            else:
                self.point_to_pick = (
                    vedo.Points(self.reference_lpoints, r=15).pickable(False).c("green")
                )
            self.at(0).add(self.point_to_pick)

            self.at(1).remove(self.other_lpoints_pointcloud)
            self.other_lpoints_pointcloud = (
                vedo.Points(self.other_lpoints, r=15).pickable(False).c("r")
            )
            self.at(1).add(self.other_lpoints_pointcloud)

    def _key_press(self, evt):
        """The _key_press method is called when the user presses a key. It is used to add or delete landmarks."""

        if self.mode == "reference" and evt.actor == self.reference:
            if evt.keypress == "z":
                pt = self.active_actor.closest_point(evt.picked3d)
                self.reference_lpoints.append(pt)

            if evt.keypress == "e":
                pt = vedo.Points(self.active_actor.points()).closest_point(evt.picked3d)
                self.reference_lpoints.append(pt)

            if evt.keypress == "d":
                if len(self.reference_lpoints) > 0:
                    self.reference_lpoints.pop()

            if evt.keypress == "s":
                self._done()

            self._update()
            self.render()

        elif self.mode == "others" and evt.actor == self.current_other:
            if evt.keypress == "z" and len(self.other_lpoints) < self.n_landmarks:
                pt = self.active_actor.closest_point(evt.picked3d)
                self.other_lpoints.append(pt)

            if evt.keypress == "e" and len(self.other_lpoints) < self.n_landmarks:
                pt = vedo.Points(self.active_actor.points()).closest_point(evt.picked3d)
                self.other_lpoints.append(pt)

            if evt.keypress == "d":
                if len(self.other_lpoints) > 0:
                    self.other_lpoints.pop()

            if evt.keypress == "s" and len(self.other_lpoints) == self.n_landmarks:
                self._done()
            else:
                self._update()

    @property
    def landmarks_as_3d_point_cloud(self):
        """Return the landmarks as a list of two lists of 3D points."""

        torch_landmarks = [
            torch.from_numpy(np.array(landmarks)).type(float_dtype)
            for landmarks in self.landmarks3d
        ]

        return torch_landmarks

    @property
    def landmarks(self, barycentric=True):
        """Return the landmarks as a list of two lists of 3D points."""

        torch_landmarks = [
            torch.from_numpy(np.array(landmarks)).type(float_dtype)
            for landmarks in self.landmarks3d
        ]
        if not barycentric:
            return torch_landmarks

        else:

            def to_coo(landmarks, n, device):
                values = torch.tensor([], dtype=float_dtype, device=device)
                indices = torch.zeros((2, 0), dtype=int_dtype, device=device)

                for i, (c, v) in enumerate(landmarks):
                    tmp = torch.concat(
                        (i * torch.ones_like(v, device=device), v)
                    ).reshape(2, -1)
                    indices = torch.cat((indices, tmp), dim=1)

                    values = torch.concat((values, c), dim=0)

                return torch.sparse_coo_tensor(
                    indices, values, (len(landmarks), n), device=device
                )

            return [
                to_coo(
                    [
                        barycentric_coordinates(self.original_meshes[i], l)
                        for l in torch_landmarks[i]
                    ],
                    self.original_meshes[i].n_points,
                    self.original_meshes[i].device,
                )
                for i in range(len(self.original_meshes))
            ]


@typecheck
def barycentric_coordinates(mesh, point):
    device = mesh.device
    point = point.to(device)

    # Compute the vectors from the point to the vertices
    vertices = mesh.points
    vectors = vertices - point
    norms = torch.norm(vectors, dim=1)

    tol = 1e-5  # TODO tol can be computed from the mesh resolution ?

    # Test if a vector is zero (that means the point is a vertex of the mesh)
    if torch.sum(vectors.abs().sum(dim=1) < tol):
        indice = torch.where(
            torch.all(torch.eq(vectors, torch.zeros_like(vectors)), dim=1)
        )[0]
        vertex_indice = indice[0]

        # The point is a vertex
        return (
            torch.tensor([1.0], device=device),
            torch.tensor([vertex_indice], device=device),
        )

    else:
        # Normalize the vectors
        vectors /= norms.reshape(-1, 1)

        A = mesh.edges[0]
        B = mesh.edges[1]

        cos_angles = (vectors[A] * vectors[B]).sum(dim=1)
        # If cos(angle) = -1 <=> angle = pi, the point is on an edge

        if torch.sum((cos_angles - (-1)).abs() < tol):
            indice = torch.where((cos_angles - (-1)).abs() < tol)[0]
            edge_indice = indice[0]

            # The point is on an edge
            # Coordinates
            a, b = mesh.edges[:, edge_indice]
            alpha = norms[b] / torch.norm(vertices[a] - vertices[b])
            beta = norms[a] / torch.norm(vertices[a] - vertices[b])

            return (
                torch.tensor([alpha, beta], device=device),
                torch.tensor([a, b], device=device),
            )

        else:
            A = mesh.triangles[0]
            B = mesh.triangles[1]
            C = mesh.triangles[2]

            angles_1 = torch.acos((vectors[A] * vectors[B]).sum(dim=1))
            angles_2 = torch.acos((vectors[B] * vectors[C]).sum(dim=1))
            angles_3 = torch.acos((vectors[C] * vectors[A]).sum(dim=1))

            sum_angles = angles_1 + angles_2 + angles_3
            # If sum_angles is close to 2pi, the point is inside the triangle, or its projection is inside the triangle

            if torch.sum((sum_angles - (2 * torch.pi)).abs() < tol):
                indices = torch.where((sum_angles - (2 * torch.pi)).abs() < tol)[0]
                # If several indices, we must find the one for which the point is inside the triangle, and not only its projection
                tmp = []
                for i in indices:
                    a, b, c = mesh.triangles[:, i]
                    normals = mesh.triangle_normals[i]

                    tmp.append(
                        torch.abs(vectors[a].dot(normals))
                        + torch.abs(vectors[b].dot(normals))
                        + torch.abs(vectors[c].dot(normals))
                    )

                indice_triangle = indices[torch.argmin(torch.tensor(tmp))]

                # The point is inside a triangle
                # Coordinates
                a, b, c = mesh.triangles[:, indice_triangle]
                mat = torch.cat((vertices[a], vertices[b], vertices[c])).reshape(3, 3).T
                alpha, beta, gamma = torch.inverse(mat) @ point

                return (
                    torch.tensor([alpha, beta, gamma], device=device),
                    torch.tensor([a, b, c], device=device),
                )

            else:
                # The point is outside the mesh
                return None
