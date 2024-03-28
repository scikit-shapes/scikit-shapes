"""
Registration with LDDMM
=======================

This examples shows the registration of two PolyDatas with the LDDMM model.
"""

# This is a fix for the documentation, you can remove it if you are running the code in a script
import sys

import pykeops

import skshapes as sks

sys.path.append(pykeops.get_build_folder())

###############################################################################
# Prepare the data
# ----------------

source = sks.PolyData("data/skulls/template.vtk")
target = sks.PolyData("data/skulls/skull_neandertalis.vtk")

# Add control points to the source shape
source.control_points = source.bounding_grid(N=50, offset=0.4)

# Add landmarks to the source and target shapes
source.landmark_indices = [94,  84, 105,  95, 106, 131, 116, 136,  68,  79,  31,  61,  47,  19]
target.landmark_indices = [49,  37,  36,  29,  13,  72,  24,   0,  59,  51, 156, 103, 123, 169]

# Define the registration model
loss = sks.LandmarkLoss() + sks.OptimalTransportLoss()
model = sks.ExtrinsicDeformation(
    n_steps=5,
    kernel="gaussian",
    scale=8,
    control_points=True
)

registration = sks.Registration(
    model=model,
    loss=loss,
    optimizer=sks.LBFGS(),
    n_iter=3,
    verbose=True,
    regularization_weight=0,
)

morphed = registration.fit_transform(source=source, target=target)

###############################################################################
# Visualize the result
# --------------------

import numpy as np
import pyvista as pv

source_color = "teal"
target_color = "red"
cpos = [(-20.266633872244565, 9.52741654099364, 653.2794560673151),
 (-25.034557342529297, 25.013988494873047, 0.0),
 (0.0009758954196487321, 0.9997188263985942, 0.023692103586369296)]

plotter = pv.Plotter(shape=(1, 2))
plotter.subplot(0, 0)
plotter.camera_position = cpos
plotter.add_mesh(source.control_points.to_pyvista(), color="black", line_width=1)
plotter.add_mesh(source.to_pyvista(), color=source_color, line_width=10)
plotter.add_mesh(target.to_pyvista(), color=target_color, line_width=10)

source_landmarks = source.landmark_points_3D.detach().cpu().numpy()
target_landmarks = target.landmark_points_3D.detach().cpu().numpy()

all_landmarks = np.concatenate([source_landmarks, target_landmarks], axis=0)
lines = []
for i in range(len(source_landmarks)):
    lines.append(2)
    lines.append(i)
    lines.append(i + len(source_landmarks))
landmarks = pv.PolyData(all_landmarks, lines=lines)
plotter.add_mesh(landmarks, color="blue", line_width=1)
plotter.add_points(landmarks.points, color="blue", point_size=10, render_points_as_spheres=True)

plotter.subplot(0, 1)
plotter.camera_position = cpos
plotter.add_mesh(morphed.control_points.to_pyvista(), color="black", line_width=1)
plotter.add_mesh(morphed.to_pyvista(), color=source_color, line_width=10)
plotter.add_mesh(target.to_pyvista(), color=target_color, line_width=10)

morphed_landmarks = morphed.landmark_points_3D.detach().cpu().numpy()
all_landmarks = np.concatenate([morphed_landmarks, target_landmarks], axis=0)
lines = []
for i in range(len(morphed_landmarks)):
    lines.append(2)
    lines.append(i)
    lines.append(i + len(morphed_landmarks))
landmarks = pv.PolyData(all_landmarks, lines=lines)
plotter.add_mesh(landmarks, color="blue", line_width=5)
plotter.add_points(landmarks.points, color="blue", point_size=10, render_points_as_spheres=True)

plotter.show()
