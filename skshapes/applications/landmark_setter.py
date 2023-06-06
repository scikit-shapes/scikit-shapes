import vedo
import numpy as np

# import pyvista as pv


class LandmarkSetter(vedo.Plotter):
    def __init__(self, meshes, **kwargs):
        super().__init__(N=2, sharecam=False, resetcam=True, **kwargs)

        self.meshes = [meshes.clone().linewidth(1).pickable(True) for meshes in meshes]

        self.reference = meshes[0]
        self.others = meshes[1:]

        self.landmarks3d = [[] for i in range(len(meshes))]

        self.reference_vertices = vedo.Points(self.reference.points())
        self.current_other = self.others[0]

        # At the beginning, the reference is active
        self.active = 0
        self.active_actor = self.reference

        self.reference_lpoints = []

        self.mode = "reference"

        text_reference = "Start by selecting landmarks on the reference mesh\nPress z to add a point on the surface\nPress e to add a vertice\nPress d to delete the last point\nPress s when you are done"
        self.reference_lpoints_pointcloud = (
            vedo.Points(self.reference_lpoints, r=15).pickable(False).c("r")
        )
        self.instructions_reference = vedo.Text2D(
            text_reference, pos="bottom-left", c="white", bg="green", font="Calco"
        )

        text_other = "Now select the same landmarks on the other meshes\nPress z to add a point on the surface\nPress e to add a vertice\nPress d to delete the last point\nPress s when you are done"
        self.instructions_other = vedo.Text2D(
            text_other, pos="bottom-left", c="white", bg="green", font="Calco"
        )

        self.add_callback("KeyPress", self._key_press)

        # At initialization, add the reference and the first other mesh to the scene
        self.at(0).add(self.instructions_reference)
        self.at(0).add(self.reference.linewidth(1))
        self.at(1).add(self.current_other.linewidth(1))

    def start(self):
        self._update()
        self.show(interactive=True)
        # return self

    def _done(self):

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
                # self.point_to_pick = vedo.Points([self.reference_lpoints[len(self.other_lpoints)]], r=15).pickable(False).c("green")
                self._update()

            else:
                self.close()

    def _update(self):

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


if __name__ == "__main__":

    meshes = [
        vedo.load("../../data/SCAPE_low_resolution/mesh00{}.ply".format(i))
        for i in range(1, 5)
    ]

    app = LandmarkSetter(meshes=meshes)
    app.start()

    print(app.landmarks3d)
