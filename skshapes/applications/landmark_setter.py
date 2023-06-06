import vedo
import numpy as np

# import pyvista as pv


class LandmarksSelector(vedo.Plotter):
    """
    Display  a mesh and allow the user to select points on it

    Args:
        mesh (vedo.Mesh): the mesh on which the points are displayed
    """

    def __init__(self, mesh, n_landmarks=None, **kwargs):
        """ """
        super().__init__(**kwargs)

        mesh = mesh.clone()  # Clone the mesh to avoid modifying the original one

        if n_landmarks is None:
            n_landmarks = 1e5

        self.n_landmarks = n_landmarks

        # Set the mesh properties
        self.mesh = mesh.linewidth(1)
        self.mesh.pickable(
            True
        )  # Make the mesh pickable to be able to select points on it
        self.vertices = vedo.Points(self.mesh.points())

        self.lpoints = []  # List of landmarks points
        self.active = True  # Flag to know if the interaction is active or not

        # Instructions
        t = "Press e to add a point on the surface\nPress z to add a vertice\nPress d to delete the last point\nPress q to quit"
        self.instructions = vedo.Text2D(
            t, pos="bottom-left", c="white", bg="green", font="Calco"
        )

        # Add the objects to the scene
        self += [self.mesh, self.instructions]

        # Callbacks
        self.callid1 = self.add_callback("KeyPress", self._key_press)

    def points(self, newpts=None):
        """Retrieve the 3D coordinates of the clicked points"""
        return np.array(self.lpoints)

    def _key_press(self, evt):

        if not self.active:
            return

        if (
            evt.keypress == "z" and evt.actor == self.mesh
        ):  # If the key pressed is z and the cursor is on the mesh
            pt = self.vertices.closest_point(
                evt.picked3d
            )  # get the closest point on the set of vertices (computed in __init__)

            self.lpoints.append(pt)  # Add the point to the list of landmarks
            if len(self.lpoints) > 1:
                self.pop()  # Remove the last set of landmarks from the scene
            # Add the new landmarks to the scene
            self.add(vedo.Points(self.lpoints, r=15).pickable(False).c("r"))

        if (
            evt.keypress == "e" and evt.actor == self.mesh
        ):  # If the key pressed is e and the cursor is on the mesh
            pt = self.mesh.closest_point(
                evt.picked3d
            )  # get the closest point on the mesh
            self.lpoints.append(pt)  # Add the point to the list of landmarks

            if len(self.lpoints) > 1:
                self.pop()  # Remove the last set of landmarks from the scene
            # Add the new landmarks to the scene
            self.add(vedo.Points(self.lpoints, r=15).pickable(False).c("r"))

        if evt.keypress == "d":
            if len(self.lpoints) > 0:  # If there are landmarks
                self.lpoints.pop()  # Remove the last landmark
                self.pop()  # Remove the last set of landmarks from the scene
                if len(self.lpoints) > 0:  # If there are still landmarks
                    self.add(
                        vedo.Points(self.lpoints, r=15).pickable(False).c("r"),
                        resetcam=False,
                    )  # Add the new landmarks to the scene

        if len(self.lpoints) == self.n_landmarks:
            self.inactivate()

    def start(self):
        """Start the interaction"""
        self.show(self.mesh, self.instructions, interactive=None)
        return self

    def inactivate(self):
        """Stop the interaction"""
        print("inactive")

        if len(self.lpoints) > 0:
            self.pop()  # Remove the points
            self.pop()  # Remove the mesh
            self.pop()  # Remove the instructions
            self.add(self.mesh)  # Add the mesh
            self.add(
                vedo.Points(self.lpoints, r=15).pickable(False).c("grey")
            )  # Add the points in grey

        else:
            self.pop()  # Remove the mesh
            self.pop()  # Remove the instructions
            self.add(self.mesh)  # Add the mesh

        self.active = False

    def highlight_point(self, point_id):

        if point_id < len(self.lpoints):
            self.clear()  # Remove everything from the scene
            self.add(self.mesh)  # Add the mesh
            self.add(
                vedo.Points(self.lpoints, r=15).c("grey")
            )  # Add the points in grey
            self.add(
                vedo.Points([self.lpoints[point_id]], r=15).c("green")
            )  # Add the selected point in red


import sys
from PyQt5 import Qt
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vedo import Plotter, Cone, load, Sphere, Points, Mesh
import pyvista as pv


class MainWindow(Qt.QMainWindow):
    def __init__(self, mesh1, mesh2, parent=None):

        Qt.QMainWindow.__init__(self, parent)
        self.frame = Qt.QFrame()
        self.layout = Qt.QGridLayout()

        # Add a horizontal layout
        self.window1 = QVTKRenderWindowInteractor(self.frame)
        self.window2 = QVTKRenderWindowInteractor(self.frame)

        # Create renderer and add the vedo objects and callbacks
        self.ls1 = LandmarksSelector(mesh=mesh1, axes=0, qt_widget=self.window1)
        self.ls1.start()
        self.empty_plotter = Plotter(qt_widget=self.window2)
        self.empty_plotter.show()

        # Set up the rest of the Qt window
        button = Qt.QPushButton("My Button makes the cone red")
        button.setToolTip("This is an example button")
        button.clicked.connect(self.onClick)
        self.layout.addWidget(self.window1, 0, 0)
        self.layout.addWidget(self.window2, 0, 1)
        self.layout.addWidget(button, 1, 0, 1, 2)
        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)
        self.show()  # NB: qt, not a Plotter method

    @Qt.pyqtSlot()
    def onClick(self):
        self.ls1.inactivate()
        self.ls1.highlight_point(0)

        n_landmarks = len(self.ls1.points())

        del self.empty_plotter
        self.ls2 = LandmarksSelector(
            mesh=mesh2, n_landmarks=n_landmarks, axes=0, qt_widget=self.window2
        )
        self.ls2.start()

        # TODO evenement pour passer au point suivant ?

    def onClose(self):
        self.window1.close()
        self.window2.close()


if __name__ == "__main__":

    mesh1 = vedo.load(vedo.dataurl + "cow.vtk")  # Load a mesh from vedo data repository

    mesh2 = mesh1.clone().rotate_x(90)  # Clone the mesh and rotate it

    # Create the landmark selector
    # ls = LandmarksSelector(mesh)
    # ls2 = LandmarksSelector(mesh2)

    app = Qt.QApplication(sys.argv)
    window = MainWindow(mesh1=mesh1, mesh2=mesh2)
    app.aboutToQuit.connect(window.onClose)
    app.exec_()
