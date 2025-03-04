"""
Set landmarks
=============

Landmarks can be set manually or using the landmark setter.
"""

import pyvista as pv
from pyvista import examples

import skshapes as sks

# sphinx_gallery_thumbnail_path = 'source/images/landmarks_skull.gif'


##############################################################################
# LandmarkSetter can be used to set landmarks interactively on a mesh.
#
# .. image:: ../../images/landmarks_skull.gif

filename = "../test_data/skulls/skull_erectus.vtk"
shape = sks.PolyData(filename)

if not pv.BUILDING_GALLERY:
    app = sks.LandmarkSetter(shape)
    app.start()
else:
    shape.landmark_indices = [95, 114, 155, 3, 9, 65, 29, 55, 74]


print(shape.landmark_indices)

##############################################################################
# Using LandmarkSetter with a list of shapes will set landmarks in correspondence.
# One the landmarks are selected on the first shape of the list (the reference shape),
# the user can select the same landmarks on the other shapes.
#
# .. image:: ../../images/demolandmarks.gif

# shape1 = sks.PolyData(examples.download_human())

shape1 = sks.PolyData(examples.download_woman())
shape2 = sks.PolyData(examples.download_doorman())

if not pv.BUILDING_GALLERY:
    app = sks.LandmarkSetter([shape1, shape2])
    app.start()
else:
    landmarks1 = [4808, 147742, 1774]
    landmarks2 = [325, 2116, 1927]
    shape1.landmark_indices = landmarks1
    shape2.landmark_indices = landmarks2

print("Landmarks shape 1:")
print(shape1.landmark_indices)
print("Landmarks shape 2:")
print(shape2.landmark_indices)
