from .vertices_landmark_setter import LandmarkSetter
from .browser import Browser

try:
    from PyQt5 import QtWidgets
    from .window_generator import WindowGenerator

except:
    pass
