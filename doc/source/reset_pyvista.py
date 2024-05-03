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
