import vedo
import numpy as np

from .._typing import *

# import pyvista as pv


class LandmarkSetter(vedo.Plotter):
    """A LandmarkSetter is a vedo application that allows the user to select landmarks on a set of meshes.

    Args:
        meshes (List[vedo.Mesh]): The meshes on which the landmarks are selected.
        **kwargs: Keyword arguments passed to the vedo.Plotter constructor.
    """

    @typecheck
    def __init__(self, meshes: List[vedo.Mesh], **kwargs) -> None:
        super().__init__(N=2, sharecam=False, **kwargs)

        # The 3D landmarks are stored in a list of lists of 3D points
        self.landmarks3d = [[] for i in range(len(meshes))]

        # Clone the meshes to avoid modifying the original ones
        self.meshes = [meshes.clone() for meshes in meshes]

        # The first mesh is the reference
        self.reference = meshes[0]
        self.others = meshes[1:]

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
        # dor display purposes
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

                print("other_id", self.other_id)

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
