# MultiScale

Multiscaling is a feature of `scikit-shapes` that allows representation of a single shape at different scales.

```python
import skshapes as sks
import pyvista.examples

bunny = sks.PolyData(pyvista.examples.download_bunny())

multiscale_bunny = sks.Multiscale(shape=bunny, ratios=[0.1, 0.01])
```
