"""Serialized functions for use in the Sphinx-Gallery configuration.

With Sphinx >= 7.3, the configuration must be serializable. This module
contains functions that can be serialized and used in the Sphinx-Gallery
configuration in conf.py.

See:
https://github.com/sphinx-gallery/sphinx-gallery/releases/tag/v0.16.0
https://github.com/sphinx-gallery/sphinx-gallery/issues/1286
"""

# Define dynamic_scraper (image scraper for pyvista)
from pyvista.plotting.utilities.sphinx_gallery import DynamicScraper

dynamic_scraper = DynamicScraper()

# Define reset_pyvista (reset pyvista module to default settings between examples)
class ResetPyVista:
    """Reset pyvista module to default settings."""

    def __call__(self, gallery_conf, fname): # noqa: ARG002
        """Reset pyvista module to default settings

        If default documentation settings are modified in any example, reset here.
        """
        import pyvista

        pyvista._wrappers['vtkPolyData'] = pyvista.PolyData
        pyvista.set_plot_theme('document')

    def __repr__(self):
        return 'ResetPyVista'

reset_pyvista = ResetPyVista()
