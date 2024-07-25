from .browser import Browser
from .vertices_landmark_setter import LandmarkSetter

try:
    from PyQt5 import QtWidgets

    from .window_generator import WindowGenerator

except ImportError:
    pass
