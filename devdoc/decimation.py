import sys
from PyQt5 import Qt
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vedo
import pyvista as pv

SURFACE_OPACITY = 1
POINTS_OPACITY = 0.5
POINTS_SIZE = 0.005


class MainWindow(Qt.QMainWindow):
    def __init__(self, filename, parent=None):

        Qt.QMainWindow.__init__(self, parent)
        self.frame = Qt.QFrame()
        self.layout = Qt.QVBoxLayout()
        self.widget = QVTKRenderWindowInteractor(self.frame)

        # Create renderer and add the vedo objects and callbacks
        self.plt = vedo.Plotter(
            N=2, axes=0, qt_widget=self.widget, sharecam=False, resetcam=True
        )

        # .linewidth(1) is equivalent to show_edges=true in pyvista
        self.mesh = vedo.load(filename).linewidth(1).alpha(SURFACE_OPACITY)
        self.mesh = (
            vedo.load(vedo.dataurl + "man.vtk").linewidth(1).alpha(SURFACE_OPACITY)
        )

        # Wrap the mesh in a pyvista object
        pv_mesh = pv.wrap(self.mesh.polydata())
        pv_mesh.triangulate(inplace=True)
        self.mesh = vedo.Mesh(pv_mesh).linewidth(1).alpha(SURFACE_OPACITY).c("grey")

        self.plt.at(0).show(
            vedo.Spheres(
                self.mesh.points(), r=POINTS_SIZE, c="blue", alpha=POINTS_OPACITY
            )
        )
        self.plt.at(1).show(
            vedo.Spheres(
                self.mesh.points(), r=POINTS_SIZE, c="blue", alpha=POINTS_OPACITY
            )
        )

        self.n_points_original = self.mesh.npoints

        self.decimate_pro_mesh = None
        self.decimate_quadric_mesh = None

        self.plt.at(0).show(self.mesh)
        self.plt.at(1).show(self.mesh)

        # Add a slider
        self.slider = Qt.QSlider(Qt.Qt.Horizontal)
        # slider.setFocusPolicy(Qt.Qt.StrongFocus)
        # slider.setTickPosition(Qt.QSlider.TicksBothSides)
        # slider.setTickInterval(10)
        # slider.setSingleStep(10)
        self.slider.setMinimum(1)
        self.slider.setMaximum(99)
        self.slider.setValue(50)

        # Set up the rest of the Qt window
        button = Qt.QPushButton("Decimate")
        button.setToolTip("This is an example button")
        button.clicked.connect(self.onClick)

        self.layout.addWidget(self.widget)
        self.layout.addWidget(self.slider)
        self.layout.addWidget(button)
        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)
        self.show()  # NB: qt, not a Plotter method

    @Qt.pyqtSlot()
    def onClick(self):
        value = self.slider.value()
        target_reduction = value / 100

        del self.decimate_pro_mesh
        del self.decimate_quadric_mesh

        self.decimate_pro_mesh = (
            self.mesh.clone()
            .decimate(target_reduction, method="pro")
            .linewidth(1)
            .alpha(SURFACE_OPACITY)
        )
        self.decimate_quadric_mesh = (
            self.mesh.clone()
            .decimate(target_reduction, method="quadric")
            .linewidth(1)
            .alpha(SURFACE_OPACITY)
        )

        n_points_pro = self.decimate_pro_mesh.npoints
        n_points_quadric = self.decimate_quadric_mesh.npoints

        actual_reduction_pro = 1 - (n_points_pro / self.n_points_original)
        actual_reduction_quadric = 1 - (n_points_quadric / self.n_points_original)

        self.plt.at(0).clear()
        self.plt.at(1).clear()

        self.plt.at(0).show(
            vedo.Spheres(
                self.mesh.points(), r=POINTS_SIZE, c="blue", alpha=POINTS_OPACITY
            ),
            resetcam=False,
        )
        self.plt.at(1).show(
            vedo.Spheres(
                self.mesh.points(), r=POINTS_SIZE, c="blue", alpha=POINTS_OPACITY
            ),
            resetcam=False,
        )
        self.plt.at(0).show(
            [
                "Pro Decimation\ntarget downsampling = {:.2f}\nactual downsampling = {:.2f}".format(
                    1 - target_reduction, actual_reduction_pro
                ),
                self.decimate_pro_mesh,
            ],
            resetcam=False,
        )
        self.plt.at(1).show(
            [
                "Quadric Decimation\ntarget downsampling = {:.2f}\nactual downsampling = {:.2f}".format(
                    1 - target_reduction, actual_reduction_quadric
                ),
                self.decimate_quadric_mesh,
            ],
            resetcam=False,
        )

    def onClose(self):
        self.widget.close()


if __name__ == "__main__":

    app = Qt.QApplication(sys.argv)
    window = MainWindow(filename="data/amygdala1.vtk")
    app.aboutToQuit.connect(window.onClose)
    app.exec_()
