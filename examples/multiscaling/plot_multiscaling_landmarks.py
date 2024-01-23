"""
Multiscaling and landmarks
==========================

Landmarks are preserved during the multiscale representation.
"""

# %%
# Imports


import pyvista
import skshapes as sks
from pyvista import examples

# %%
# Load a mesh and define landmarks

mesh = sks.PolyData(examples.download_louis_louvre().clean())

landmarks = [
    151807,
    21294,
    23344,
    25789,
    131262,
    33852,
    171465,
    191680,
    172653,
    130895,
    9743,
    19185,
    143397,
    200885,
]

mesh.landmark_indices = landmarks

# %%
# Create the multiscale representation

ratios = [0.1, 0.01, 0.001]
multiscale = sks.Multiscale(shape=mesh, ratios=ratios)


# %%
# Visualize

cpos = [
    (11.548847281353684, -19.784217817604652, 8.581378858601008),
    (1.606399655342102, 2.120710074901581, 8.925199747085571),
    (-0.021305501811855157, -0.025357814432988603, 0.9994513779267742),
]

p = pyvista.Plotter()
p.enable_hidden_line_removal()
p.open_gif("animation.gif", fps=2)
for ratio in [1, *ratios]:
    p.clear_actors()
    p.add_mesh(
        multiscale.at(ratio=ratio).to_pyvista(),
        color="tan",
        opacity=0.99,
        show_edges=True,
    )
    p.add_points(
        multiscale.at(ratio=ratio).landmark_points.numpy(),
        color="red",
        point_size=20,
        render_points_as_spheres=True,
    )

    p.camera_position = cpos
    p.write_frame()

p.show()
