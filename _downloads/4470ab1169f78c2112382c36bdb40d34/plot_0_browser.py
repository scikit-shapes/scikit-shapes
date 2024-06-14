"""
Browse a sequence of shapes
===========================

A Browser can be used to visualize a sequence of shapes. The user can navigate
through the sequence via a slider.
"""

import pyvista as pv

import skshapes as sks

# sphinx_gallery_thumbnail_path = 'source/images/demo_browser.gif'

source = sks.PolyData("../data/cactus/cactus3.ply")
target = sks.PolyData("../data/cactus/cactus11.ply")

loss = sks.L2Loss()
model = sks.IntrinsicDeformation(n_steps=5)

registration = sks.Registration(
    model=model,
    loss=loss,
    optimizer=sks.LBFGS(),
    n_iter=1,
    regularization_weight=0,
)

registration.fit(source=source, target=target)

###############################################################################
# Browser
#
# if you are running this script locally, the vedo window containing the
# browser will be displayed. The, you can navigate through the sequence of
# shapes using the slider and adjust the camera position.
#
# .. image:: ../../images/demo_browser.gif

if not pv.BUILDING_GALLERY:
    app = sks.Browser(registration.path_)
    app.show()
