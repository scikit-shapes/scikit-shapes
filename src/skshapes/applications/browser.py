"""Wrapper around [`vedo.applications.Browser`](https://vedo.embl.es/docs/vedo/applications.html#Browser)."""

import vedo

from ..types import shape_type


class Browser:
    """Application to browse a sequence of shapes with a slider.

    Based on [`vedo.applications.Browser`](https://vedo.embl.es/docs/vedo/applications.html#Browser).

    Parameters
    ----------
    shapes
        The shapes to visualize.

    Examples
    --------
    You can visualize any sequence of shapes with a slider to navigate through:

    .. code-block:: python

        import skshapes as sks

        # Create a sequence of translated spheres to visualize
        meshes = [sks.Sphere() for _ in range(5)]
        for i in range(5):
            meshes[i].points += torch.tensor([i / 5, 0, 0])
        # Create a browser to visualize the sequence
        browser = sks.Browser(meshes)
        browser.show()

    This application is useful to visualize the intermediate shapes in a
    registration task:

    .. code-block:: python

        import skshapes as sks
        import torch

        source, target = sks.Sphere(), sks.Sphere()
        # Translate the target
        target.points += torch.tensor([1, 0, 0])
        # Create a registration task with a rigid motion model and L2 loss
        model = sks.RigidMotion(n_steps=5)
        loss = sks.L2Loss()
        task = sks.Registration(model=model, loss=loss, n_iter=5)
        task.fit(source=source, target=target)
        # Visualize the intermediate shapes
        path = task.path_
        browser = sks.Browser(path)
        browser.show()

    """

    def __init__(self, shapes: list[shape_type]):
        vedo_shapes = [shape.to_vedo() for shape in shapes]
        self.vedo_browser = vedo.applications.Browser(vedo_shapes)

    def show(self):
        """Show the browser."""
        self.vedo_browser.show()
