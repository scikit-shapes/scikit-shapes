import pyvista
import numpy as np

# necessary when building the sphinx gallery
pyvista.BUILDING_GALLERY = True
pyvista.OFF_SCREEN = True
scraper_pv = pyvista._get_sg_image_scraper()

# Optional - set parameters like theme or window size
# pyvista.set_plot_theme('document')
# pyvista.global_theme.window_size = np.array([1024, 768]) * 2

extensions = [
    "mkdocs_gallery.gen_gallery",
    "pyvista.ext.viewer_directive",
]

conf = {
    "image_scrapers": ("matplotlib", scraper_pv),
}
