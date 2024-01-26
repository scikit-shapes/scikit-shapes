"""Landmark setter application."""

from typing import Union

import torch
import vedo

from ..input_validation import typecheck
from ..types import polydata_type


class LandmarkSetter:
    """Landmark setter for a PolyData or a list of PolyData.

    It could be used to select landmarks on a single shape or on a set of
    shapes.

    If a list is provided, the first shape is considered as the reference shape
    and the landmarks are selected on this shape. Then, the same landmarks must
    be selected on the other shapes.

    Parameters
    ----------
    shapes
        The shape or list of shapes on which the landmarks are selected.
    """

    @typecheck
    def __init__(
        self, shapes: Union[list[polydata_type], polydata_type]
    ) -> None:
        """Class constructor."""
        super().__init__()

        if hasattr(shapes, "__iter__") and len(shapes) > 1:
            self.landmarks_setter = LandmarkSetterMultipleMeshes(shapes)
        elif hasattr(shapes, "__iter__") and len(shapes) == 1:
            self.landmarks_setter = LandmarkSetterSingleMesh(shapes[0])
        else:
            self.landmarks_setter = LandmarkSetterSingleMesh(shapes)

    def start(self):
        """Start the landmark setter."""
        self.landmarks_setter.start()


class LandmarkSetterSingleMesh(vedo.Plotter):
    """Single PolyData landmark setter.

    Parameters
    ----------
    shape
        The PolyData on which the landmarks are selected.
    """

    def __init__(self, shape: polydata_type) -> None:
        super().__init__()

        self.shape = shape
        self.landmarks = []

        self.actor = self.shape.to_vedo().linewidth(1)
        self.actor.pickable(True)

        self.add(self.actor)

        self.lpoints = []
        self.lpoints_pointcloud = (
            vedo.Points(self.lpoints, r=15).pickable(False).c("r")
        )
        self.add(self.lpoints_pointcloud)
        self._closed = False

        text = (
            "Start by selecting landmarks on the reference shape\n"
            "Press e to add a vertice\nPress d to delete the last point\n"
            "Press z to validate the landmarks and close the window"
        )
        self.instructions = vedo.Text2D(
            text, pos="bottom-left", c="white", bg="green", font="Calco"
        )
        self.add(self.instructions)

        self.add_callback("KeyPress", self._key_press)

    def _key_press(self, evt):
        """React to a key press.

        It is used to add or delete landmarks and update the display.
        """
        if evt.keypress == "e" and evt.picked3d is not None:
            pt = vedo.Points(self.actor.vertices).closest_point(evt.picked3d)
            indice = closest_vertex(self.actor.vertices.copy(), pt)
            self.lpoints.append(pt)
            self.landmarks.append(indice)

        if evt.keypress == "d" and len(self.lpoints) > 0:
            self.lpoints.pop()
            self.landmarks.pop()

        if evt.keypress == "z" and len(self.lpoints) > 0:
            # Store the landmarks in the shape
            self.shape.landmark_indices = torch.tensor(self.landmarks)
            # Close the window
            self.close()
            self._closed = True

        if not self._closed:
            # Update the display
            self.remove(self.lpoints_pointcloud)
            self.lpoints_pointcloud = (
                vedo.Points(self.lpoints, r=15).pickable(False).c("r")
            )
            self.add(self.lpoints_pointcloud)
            self.render()

    def start(self):
        """Start the landmark setter."""
        self.reset_camera()
        self.show(interactive=True)


class LandmarkSetterMultipleMeshes(vedo.Plotter):
    """Multiple PolyData landmark setter.

    The same number of landmarks must be selected on each shape, and the
    landmarks must be selected in the same order on each shape, in order
    to use the landmarks for registration between these.
    """

    @typecheck
    def __init__(self, shapes: list[polydata_type]) -> None:
        """Class constructor.

        Parameters
        ----------
        shapes
            The list of PolyData on which the landmarks are selected.
        """
        super().__init__(N=2, sharecam=False)

        # The landmarks (list of indices) are stored in a list of lists of
        # indices
        self.landmarks = [[] for i in range(len(shapes))]

        # Convert the shapes to vedo.Mesh objects
        self.shapes = shapes
        self.objects = [shape.to_vedo() for shape in shapes]

        # The first actor is the reference
        self.reference = self.objects[0]
        self.others = self.objects[1:]

        # At the beginning : the reference shape is plotted on the left
        # and the first other shape is plotted on the right
        # TODO : set a better camera position
        self.current_other = self.others[0]
        self.at(0).add(self.reference.linewidth(1))
        self.at(1).add(self.current_other.linewidth(1))

        # At the beginning, we are in 'reference' mode, meaning that we are
        # selecting landmarks on the reference shape
        self.active_actor = self.reference
        self.reference_lpoints = []
        self.reference_indices = []
        # The reference landmarks are stored in a vedo.Points object
        # for display purposes
        self.reference_lpoints_pointcloud = (
            vedo.Points(self.reference_lpoints, r=15).pickable(False).c("r")
        )
        self.mode = "reference"
        # The reference vertices are stored in a vedo.Points object, we do not
        # display them but we store them to be able to
        # pick them
        self.reference_vertices = vedo.Points(self.reference.vertices)

        # Instructions corresponding to the "reference" mode
        text_reference = (
            "Start by selecting landmarks on the reference shape"
            "\nPress e to add a vertice\nPress d to delete the last point\n"
            "Press z to validate the landmarks"
        )
        self.instructions_reference = vedo.Text2D(
            text_reference,
            pos="bottom-left",
            c="white",
            bg="green",
            font="Calco",
        )
        self.at(0).add(
            self.instructions_reference
        )  # Add the instructions to the left plot

        # Instructions corresponding to the "other" mode (not displayed at the
        # beginning)
        text_other = (
            "Now select the same landmarks on the other shapes\n"
            "Press e to add a vertice\nPress d to delete the last point\n"
            "Press z when you have selected all the landmarks"
        )
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
        self.show(interactive=True, resetcam=False)

    def _done(self):
        """Move to the next mode or close the window.

        The _done method is called when the user presses the 's' key.
        If the current mode is 'reference', it stores information about the
            number of landmarks to be set on the other shapes and switches to
            'others' mode.
        If the current mode is 'others', it stores the landmarks for the
            current shape and switches to the next shape. If no other shape is
            left, it closes the window.
        """
        if self.mode == "reference":
            self.n_landmarks = len(self.reference_lpoints)
            self.mode = "others"
            self.at(0).remove(self.instructions_reference)
            self.at(1).add(self.instructions_other)
            self.other_id = 0
            self.other_lpoints = []
            self.other_indices = []

            self.current_other = self.others[self.other_id]

            self.active_actor = self.current_other
            self.other_lpoints_pointcloud = (
                vedo.Points(self.other_lpoints, r=15).pickable(False).c("r")
            )
            self.at(0).remove(self.reference_lpoints_pointcloud)
            self.reference_lpoints_pointcloud.c("grey")
            self.point_to_pick = (
                vedo.Points(
                    [self.reference_lpoints[len(self.other_lpoints)]], r=15
                )
                .pickable(False)
                .c("green")
            )
            self.at(0).add(self.reference_lpoints_pointcloud)

            self.landmarks[0] = self.reference_indices
            self._update()

        else:
            self.landmarks[self.other_id + 1] = self.other_indices.copy()
            self.other_id += 1
            self.at(1).clear()

            if self.other_id < len(self.others):
                self.other_lpoints = []
                self.other_indices = []
                self.current_other = self.others[self.other_id]
                self.active_actor = self.current_other

                self.at(1).add(self.current_other.linewidth(1))

                self.other_lpoints_pointcloud = (
                    vedo.Points(self.other_lpoints, r=15)
                    .pickable(False)
                    .c("r")
                )
                self._update()

            else:
                ls = self.landmarks
                for i in range(len(ls)):
                    self.shapes[i].landmark_indices = torch.tensor(ls[i])

                self.close()

    def _update(self):
        """Update the plot.

        The _update method update the display of the landmarks with the
        right color depending on the current mode and the current state of the
        landmarks selection.
        """
        if self.mode == "reference":
            self.at(0).remove(self.reference_lpoints_pointcloud)
            self.reference_lpoints_pointcloud = (
                vedo.Points(self.reference_lpoints, r=15)
                .pickable(False)
                .c("r")
            )
            self.at(0).add(self.reference_lpoints_pointcloud)

        else:
            self.at(0).remove(self.point_to_pick)
            if len(self.other_lpoints) < self.n_landmarks:
                self.point_to_pick = (
                    vedo.Points(
                        [self.reference_lpoints[len(self.other_lpoints)]], r=15
                    )
                    .pickable(False)
                    .c("green")
                )
            else:
                self.point_to_pick = (
                    vedo.Points(self.reference_lpoints, r=15)
                    .pickable(False)
                    .c("green")
                )
            self.at(0).add(self.point_to_pick)

            self.at(1).remove(self.other_lpoints_pointcloud)
            self.other_lpoints_pointcloud = (
                vedo.Points(self.other_lpoints, r=15).pickable(False).c("r")
            )
            self.at(1).add(self.other_lpoints_pointcloud)

    def _key_press(self, evt):
        """React to a key press.

        The _key_press method is called when the user presses a key. It is
        used to add or delete landmarks.
        """
        if self.mode == "reference" and evt.actor == self.reference:
            if evt.keypress == "e" and evt.picked3d is not None:
                pt = vedo.Points(self.active_actor.vertices).closest_point(
                    evt.picked3d
                )
                indice = closest_vertex(self.active_actor.vertices.copy(), pt)
                self.reference_lpoints.append(pt)
                self.reference_indices.append(indice)

            if evt.keypress == "d" and len(self.reference_lpoints) > 0:
                self.reference_lpoints.pop()
                self.reference_indices.pop()

            if evt.keypress == "z" and len(self.reference_lpoints) > 0:
                self._done()

            self._update()

        elif self.mode == "others" and evt.actor == self.current_other:
            if (
                evt.keypress == "e"
                and len(self.other_lpoints) < self.n_landmarks
            ) and evt.picked3d is not None:
                pt = vedo.Points(self.active_actor.vertices).closest_point(
                    evt.picked3d
                )
                indice = closest_vertex(self.active_actor.vertices.copy(), pt)
                self.other_lpoints.append(pt)
                self.other_indices.append(indice)

            if evt.keypress == "d" and len(self.other_lpoints) > 0:
                self.other_lpoints.pop()
                self.other_indices.pop()

            if (
                evt.keypress == "z"
                and len(self.other_lpoints) == self.n_landmarks
            ):
                self._done()
            else:
                self._update()

        self.render()


@typecheck
def closest_vertex(points, point):
    """Clostest vertex computation.

    Given a list of vertices and a point, return the indice of the closest
    vertex.
    """
    # Compute the vectors from the point to the vertices
    vertices = torch.tensor(points)
    point = torch.tensor(point)
    vectors = vertices - point
    norms = torch.norm(vectors, dim=1)

    tol = 1e-5  # TODO tol can be computed from the shape resolution ?

    # Test if a vector is zero (that means the point is a vertex of the shape)
    if torch.sum(vectors.abs().sum(dim=1) < tol):
        indice = torch.where(
            torch.all(torch.eq(vectors, torch.zeros_like(vectors)), dim=1)
        )[0]
        vertex_indice = indice[0]

        # The point is a vertex
        return int(vertex_indice)

    else:
        # Error if the point is not on the shape
        if torch.sum(norms < tol) == 0:
            msg = "The point is not a vertex of the shape"
            raise ValueError(msg)
        return None
