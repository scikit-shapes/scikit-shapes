import sys
from PyQt5 import Qt, QtWidgets
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vedo import Plotter, Sphere, Cone


class WindowGenerator(Qt.QMainWindow):
    """A window with custom widgets: buttons, checkboxes, vedoPlotter...

    Args:
        widgets (list): A list of dictionaries, each dictionary should have the following keys:
            - type: The type of the widget, can be one of: "CheckBox", "Choice", "Button", "Label", "VTKPlotter"
            - key: A unique key to identify the widget
            - value: The value of the widget, depends on the type of the widget:
                - CheckBox: The text to display next to the checkbox
                - Choice: A list of choice values
                - Button: The text to display on the button
                - Label: The text to display in the label
                - VTKPlotter: A vedo object constructor like Sphere or Cone or None (if no object to display)
    """

    def __init__(self, widgets):
        self.app = QtWidgets.QApplication(sys.argv)
        super().__init__(parent=None)

        self.central_widget = QtWidgets.QWidget(self)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)
        self.widgets = {}  # Dictionary to hold the widget objects

        for widget in widgets:
            self.add_widget(widget)

        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

    def add_widget(self, widget):
        widget_type = widget["type"]
        key = widget["key"]
        value = widget["value"]

        if widget_type == "CheckBox":
            checkbox = QtWidgets.QCheckBox(value)
            self.layout.addWidget(checkbox)
            self.widgets[key] = checkbox

        elif widget_type == "Choice":
            combobox = QtWidgets.QComboBox()
            combobox.addItems(value)
            self.layout.addWidget(combobox)
            self.widgets[key] = combobox

        elif widget_type == "EditableText":
            textedit = QtWidgets.QLineEdit()
            textedit.setPlaceholderText(value)
            self.layout.addWidget(textedit)
            self.widgets[key] = textedit

        elif widget_type == "Button":
            button = QtWidgets.QPushButton(value)
            self.layout.addWidget(button)
            self.widgets[key] = button

        elif widget_type == "Label":
            label = QtWidgets.QLabel(value)
            self.layout.addWidget(label)
            self.widgets[key] = label

        elif widget_type == "VTKPlotter":
            vtkWidget = QVTKRenderWindowInteractor(self.central_widget)
            plotter = Plotter(qt_widget=vtkWidget)
            plotter += value  # value should be a vedo object
            plotter.show()
            self.layout.addWidget(vtkWidget)
            self.widgets[key] = {"vtkWidget": vtkWidget, "plotter": plotter}

    def add_callback(self, key, callback):
        assert key in self.widgets, f"No widget found with key: {key}"
        widget = self.widgets[key]
        assert isinstance(
            widget, QtWidgets.QPushButton
        ), f"The widget with key {key} is not a button"

        widget.clicked.connect(callback)

    def show(self):
        super().show()
        self.app.exec_()
