"""
Landmark setter example
=======================

In this example, we demonstrate how to use the landmark setter application.
"""

# %%
# ## Load data
#

from pyvista import examples
a = 1
import skshapes as sks

shape1 = sks.PolyData(examples.download_human())
shape2 = sks.PolyData(examples.download_doorman())
shape1.point_data.clear()
shape2.point_data.clear()

# %%
# ## Run the application
#
# Here the line app.start() is commented out because it would block the
# execution of the notebook. Uncomment it to run the application in a script.
# Below there is a screen recording of the application in action.


# app = sks.LandmarkSetter([shape1, shape2])
# app.start()

# %%
# ![](../../../imgs/demolandmarks.gif)
#
# Application: now, shape1 and shape2 have landmarks, you can access them with
# shape1.landmark_indices and shape2.landmark_indices. You can use them with
# the landmark loss.
