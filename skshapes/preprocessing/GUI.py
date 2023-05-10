"""This module contains the GUI for the skshapes program.
"""
import os
os.environ["QT_API"] = "pyqt5"


from qtpy import QtWidgets
import pyvista as pv
from pyvistaqt import QtInteractor, MainWindow
from typing import Optional, Tuple
import numpy as np
import yaml
from enum import Enum

class ShapesViewer(MainWindow):
    def __init__(
        self,
        parent: Optional[MainWindow] = None,
        title: Optional[str] = None,
        size: Optional[Tuple[int, int]] = None,
        show: Optional[bool] = False,
    ) -> None:
        super().__init__(parent, title, size)

        # Set the menu bar
        mainMenu = self.menuBar()

        # File menu
        fileMenu = mainMenu.addMenu("File")

        # Exit button
        exitButton = QtWidgets.QAction("Exit", self)
        exitButton.setShortcut("Ctrl+Q")
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)

        # Load new files button
        loadButton = QtWidgets.QAction("Load new Files", self)
        loadButton.triggered.connect(self.load_files)
        fileMenu.addAction(loadButton)

        # Edit menu
        editMenu = mainMenu.addMenu("Edit")
        # Undo button
        undoButton = QtWidgets.QAction("Undo", self)
        undoButton.setShortcut("Ctrl+Z")
        editMenu.addAction(undoButton)
        # Redo button
        redoButton = QtWidgets.QAction("Redo", self)
        redoButton.setShortcut("Ctrl+Y")
        editMenu.addAction(redoButton)

        # Define the layouts
        # #############################
        # # topLayout (buttons) #######
        # #############################
        # # leftLayout ### rightLayout#
        # # (list of   ### (plotter)  #
        # #   files)   ###            #
        # #############################

        # Create the main widget and layout
        centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(centralWidget)
        mainLayout = QtWidgets.QVBoxLayout(centralWidget)
        # Create top layout for buttons
        topLayout = QtWidgets.QHBoxLayout()
        mainLayout.addLayout(topLayout)
        # Create horizontal layout for file selection and content display
        bottomLayout = QtWidgets.QHBoxLayout()
        mainLayout.addLayout(bottomLayout)
        # Create left layout for file selection
        leftLayout = QtWidgets.QVBoxLayout()
        bottomLayout.addLayout(leftLayout, stretch=3)
        # Create right layout for content display
        rightLayout = QtWidgets.QVBoxLayout()
        bottomLayout.addLayout(rightLayout, stretch=7)

        # Add widgets to the layouts

        # Create buttons
        self.alignButton = QtWidgets.QPushButton("Align")
        self.setLandmarksButton = QtWidgets.QPushButton("Set Landmarks")
        self.saveSceneButton = QtWidgets.QPushButton("Save Scene")
        self.saveSceneButton.clicked.connect(self.saveScene)
        self.chooseLandmarksType = QtWidgets.QComboBox()
        self.chooseLandmarksType.addItem("No landmarks")
        self.chooseLandmarksType.addItem("Landmarks on reference shape")
        self.chooseLandmarksType.addItem("Landmarks on all shapes")
        self.chooseLandmarksType.setCurrentIndex(0)
        topLayout.addWidget(self.chooseLandmarksType)
        topLayout.addWidget(self.alignButton)
        topLayout.addWidget(self.setLandmarksButton)
        topLayout.addWidget(self.saveSceneButton)

        # Create file list widget
        self.fileList = QtWidgets.QListWidget()
        leftLayout.addWidget(self.fileList, stretch=1)

        # Create frame for content display
        self.frame = QtWidgets.QFrame()
        self.plotter = QtInteractor(self.frame)
        self.signal_close.connect(self.plotter.close)
        rightLayout.addWidget(self.plotter)

        # Initialize scene
        self.chooseLandmarksType.currentIndexChanged.connect(self.print_landmarks_type)
        self.scene = Scene(self.plotter)
        self.alignButton.clicked.connect(self.scene.do_rigid_alignment)
        self.setLandmarksButton.clicked.connect(self.set_landmarks)
        undoButton.triggered.connect(self.scene.undo_last_transformation)

        if show:
            self.show()

    def print_landmarks_type(self):

        print(self.chooseLandmarksType.currentText())


    def load_files(self):
        """Load new files into the program."""
        # Open file dialog to select files
        fileNames, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Load Files", "", "All files (*.*)"
        )

        # Clear file list widget
        self.fileList.clear()

        # Add selected files to file list widget
        for fileName in fileNames:

            item = QtWidgets.QListWidgetItem(fileName.split("/")[-1])
            item.setCheckState(False)
            self.fileList.addItem(item)

        # Update scene
        self.scene.SetNewFiles(fileNames, self.fileList)

        self.fileList.itemChanged.connect(
            lambda: self.scene.updateVisibilities(self.fileList)
        )

    def set_landmarks(self):
        """Set landmarks for the scene."""
        # Inactivate current window
        self.setEnabled(False)
        if self.scene.n_shapes > 0:
            LandmarkSelector(scene=self.scene, parent=self)

        # Reactivate current window
        self.setEnabled(True)

    def saveScene(self):
        """Save the scene."""
        # Open file dialog to select files
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Scene", "", "All files (*.*)"
        )
        # Save scene
        self.scene.save(fileName)


class LandmarkSelector(MainWindow):

    """This class is used to select landmarks for the scene (collection of shapes).
    First, the button add landmark is connected to the function to add a landmark on the reference shape.
    Then, after the button Done is pressed, the user can select landmarks on the other shapes.
    At the end of the process the landmarks are saved in the scene object.
    """

    def __init__(
        self,
        scene,
        parent: Optional[MainWindow] = None,
        title: Optional[str] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__(parent, title, size)

        self.scene = scene
        self.landmarks = []

        # Layout

        # Create the main widget and layout
        centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(centralWidget)
        mainLayout = QtWidgets.QVBoxLayout(centralWidget)

        # Create an horizontal layout for the buttons
        buttonLayout = QtWidgets.QHBoxLayout()
        mainLayout.addLayout(buttonLayout)

        # Create an horizontal bottom layout with two plotters
        centralLayout = QtWidgets.QHBoxLayout()
        mainLayout.addLayout(centralLayout)

        # Create a bottom layout for instructions
        bottomLayout = QtWidgets.QHBoxLayout()
        mainLayout.addLayout(bottomLayout)

        self.frame = QtWidgets.QFrame()

        # create the left plotter
        self.plotter1 = QtInteractor(self.frame)
        centralLayout.addWidget(self.plotter1)

        # create the right plotter
        self.plotter2 = QtInteractor(self.frame)
        centralLayout.addWidget(self.plotter2)

        # Add buttons
        self.addLanmdarkButton = QtWidgets.QPushButton("Add landmark (Ctrl+Enter)")
        buttonLayout.addWidget(self.addLanmdarkButton)

        self.removeLandmarkButton = QtWidgets.QPushButton("Remove landmark")
        self.removeLandmarkButton.clicked.connect(self.remove_landmark)
        buttonLayout.addWidget(self.removeLandmarkButton)

        # Add text to bottom layout
        self.instructionsText = QtWidgets.QLineEdit(
            "Add landmarks to the reference shape, press Done when finished"
        )
        self.instructionsText.setReadOnly(True)
        bottomLayout.addWidget(self.instructionsText)
        self.DoneButton = QtWidgets.QPushButton("Done")
        bottomLayout.addWidget(self.DoneButton)

        # Initialize the left plotter with the reference shape
        self.reference_shape = scene.meshes[0]
        self.plotter1.add_mesh(self.reference_shape)
        self.plotter1.enable_point_picking(
            pickable_window=False, show_message=False, use_mesh=True, callback=None
        )
        self.reference_landmarks = []

        self.addLanmdarkButton.clicked.connect(
            lambda: self.add_landmark(
                plotter=self.plotter1,
                shape=self.reference_shape,
                landmarks=self.reference_landmarks,
            )
        )
        self.DoneButton.clicked.connect(self.reference_landmarks_done)

        self.show()

    def add_landmark(self, plotter, shape, landmarks):

        if plotter.picked_point is None:
            pass
        else:
            index = ((shape.points - plotter.picked_point) ** 2).sum(axis=1).argmin()
            landmarks.append(index)
            plotter.add_points(
                shape.points[index],
                color="red",
                point_size=10,
                reset_camera=False,
                render_points_as_spheres=True,
            )

    def reference_landmarks_done(self):
        """When the user is done selecting landmarks on the reference shape, the current shape index is set to 1
        and the select_landmarks_on_a_new_shape function is called.
        """

        self.landmarks.append(self.reference_landmarks)
        self.n_landmarks = len(self.reference_landmarks)
        self.current_shape_index = 1

        self.select_landmarks_on_a_new_shape()

    def select_landmarks_on_a_new_shape(self):

        if self.current_shape_index == len(self.scene.meshes):
            # If all the shapes have been processed, save the landmarks and close the window
            self.scene.landmarks = self.landmarks
            self.close()

        else:

            # If there are still shapes to process, reinitialize the active landmarks and the active shape
            self.active_landmarks = []
            self.active_shape = self.scene.meshes[self.current_shape_index]

            # Update the instructions text
            self.updateInstructionsText()

            # Reinitialize the left plotter with the reference shape and the landmarks in grey
            self.plotter1.disable_picking()
            self.plotter1.clear_actors()
            self.plotter1.add_mesh(self.reference_shape, reset_camera=False)
            self.plotter1.add_points(
                self.reference_shape.points[self.reference_landmarks],
                color="grey",
                point_size=10,
                reset_camera=False,
                render_points_as_spheres=True,
            )

            # Highlight the first landmark in green
            referenceLandmark = self.reference_landmarks[0]
            self.plotter1.add_points(
                self.reference_shape.points[referenceLandmark],
                reset_camera=False,
                render_points_as_spheres=True,
                color="green",
                point_size=10,
            )

            # Initialize the right plotter with the active shape
            self.plotter2.clear_actors()
            self.plotter2.add_mesh(self.active_shape)
            self.plotter2.enable_point_picking(
                pickable_window=False, show_message=False
            )

            # Connect the add Landmark button to the make_correspondence method
            self.addLanmdarkButton.clicked.disconnect()
            self.addLanmdarkButton.clicked.connect(self.make_correspondence)

    def make_correspondence(self):

        if self.plotter2.picked_point is not None:

            index = (
                ((self.active_shape.points - self.plotter2.picked_point) ** 2)
                .sum(axis=1)
                .argmin()
            )
            self.active_landmarks.append(index)
            self.plotter2.add_points(
                self.plotter2.picked_point,
                color="red",
                point_size=10,
                reset_camera=False,
                render_points_as_spheres=True,
            )

            self.updateInstructionsText()

            if len(self.active_landmarks) == self.n_landmarks:
                self.current_shape_index += 1
                self.landmarks.append(self.active_landmarks)
                self.select_landmarks_on_a_new_shape()
            else:
                oldReferenceLandmark = self.reference_landmarks[
                    len(self.active_landmarks) - 1
                ]
                self.plotter1.add_points(
                    self.reference_shape.points[oldReferenceLandmark],
                    color="red",
                    point_size=10,
                    reset_camera=False,
                    render_points_as_spheres=True,
                )
                newReferenceLandmark = self.reference_landmarks[
                    len(self.active_landmarks)
                ]
                self.plotter1.add_points(
                    self.reference_shape.points[newReferenceLandmark],
                    color="green",
                    point_size=10,
                    reset_camera=False,
                    render_points_as_spheres=True,
                )

    def remove_landmark(self):
        # TODO
        pass

    def updateInstructionsText(self):

        self.instructionsText.clear()
        self.instructionsText.insert(
            "Set landmark {}/{} on the shape {}/{}".format(
                len(self.active_landmarks) + 1,
                self.n_landmarks,
                self.current_shape_index + 1,
                len(self.scene.meshes),
            )
        )


class Scene:
    """A scene is a collection of shapes with display options.
    It is initialized with a a list of file names through the SetNewFiles method.
    Then, it provides display methods to update the scene in the GUI and preprocessing
    steps recorded in a history.
    """

    def __init__(self, plotter: QtInteractor) -> None:

        self.plotter = plotter
        
        self.actors = []
        self.visibilities = []
        self.meshes = []
        self.landmarks = []
        self.files = []
        self.transformations = []
        self.n_shapes = 0

    def SetNewFiles(self, fileNames: list, fileList: QtWidgets.QListWidget) -> None:
        """Initialize actors, visibilities, mesh for new files."""
        # Reset files
        self.files = fileNames
        self.n_shapes = len(fileNames)

        # Reset visibilities
        self.visibilities = [False] * len(fileNames)
        # Update visibilities
        for index in range(fileList.count()):
            self.visibilities[index] = bool(fileList.item(index).checkState())

        # Initialize meshes
        self.meshes = [pv.read(file) for file in fileNames]

        # Remove old actors
        for actor in self.actors:
            self.plotter.remove_actor(actor)

        # Define new actors
        self.actors = [self.plotter.add_mesh(mesh, opacity=0.9) for mesh in self.meshes]
        for actor, visibility in zip(self.actors, self.visibilities):
            actor.SetVisibility(visibility)

    def do_rigid_alignment(self):
        """Do rigid alignment of meshes."""
        transformation = RigidTransformation(landmarks_type="all", reference_index=0)
        self.meshes = transformation.apply(self.meshes)
        self.transformations.append(transformation)
        self.updateActors()

    def undo_last_transformation(self):
        """Undo last transformation."""

        # If there is no transformation to undo, do nothing
        if len(self.transformations) == 0:
            pass

        # Otherwise, undo last transformation
        else:
            last_transformation = self.transformations.pop()
            self.meshes = last_transformation.undo(self.meshes)
            self.updateActors()

    def updateVisibilities(self, fileList: QtWidgets.QListWidget) -> None:
        """Update the plotter with the current scene."""
        # Update visibilities
        for index in range(fileList.count()):
            self.visibilities[index] = bool(fileList.item(index).checkState())

        for actor, visibility in zip(self.actors, self.visibilities):
            actor.SetVisibility(visibility)

    def updateActors(self) -> None:
        """Update the plotter with the current scene."""
        for i in range(len(self.actors)):
            self.plotter.remove_actor(self.actors[i])
            self.actors[i] = self.plotter.add_mesh(self.meshes[i], opacity=0.8)
            self.actors[i].SetVisibility(self.visibilities[i])

    def save(self, fileName: str) -> None:
        """Save the scene parameters in a .yaml file."""

        data = dict(
            files=[file for file in self.files],
            landmarks=[
                [int(landmark) for landmark in landmarks]
                for landmarks in self.landmarks
            ],
            rigid_transformation=dict(reference_shape=0, apply_to="all"),
        )

        with open(fileName, "w") as outfile:
            yaml.safe_dump(data, outfile, default_flow_style=None, sort_keys=False)

        # TODO
        pass


# from meshes import rigid_interpolation


# class RigidTransformation:
#     """
#     A transformation is initialized with a list of hyperparameters
#     it is then applied to a scene and recorded in the history
#     apply takes a list of meshes and returns a list of meshes (the transformed meshes)
#     undo returns the meshes, before the transformation was applied, the simpler option is to store the meshes before the transformation
#     we can also store the transformation parameters and apply the inverse transformation if applicable
#     """

#     def __init__(
#         self, landmarks_type: Optional[str] = "all", reference_index: Optional[int] = 0
#     ) -> None:
#         self.landmarks_type = landmarks_type
#         self.reference_index = reference_index

#     def apply(self, meshes: list) -> list:

#         self.previous_meshes = meshes
#         reference_mesh = meshes[self.reference_index]
#         transformed_meshes = []
#         for i in range(len(meshes)):
#             if i != self.reference_index:
#                 alpha = rigid_interpolation(meshes[i], reference_mesh)
#                 transformed_meshes.append(alpha(1))

#             else:
#                 transformed_meshes.append(meshes[i])

#         return transformed_meshes

#     def undo(self, meshes: list) -> list:

#         return self.previous_meshes


if __name__ == "__main__":
    import sys

    if not os.path.exists("tmp_run"):
        os.makedirs("tmp_run")
    # change environment variable XDG_RUNTIME_DIR to run directory
    os.environ["XDG_RUNTIME_DIR"] = "tmp_run"

    app = QtWidgets.QApplication(sys.argv)
    window = ShapesViewer(title="Preprocessing Viewer", show=True)
    os.rmdir("tmp_run")
    sys.exit(app.exec_())

    