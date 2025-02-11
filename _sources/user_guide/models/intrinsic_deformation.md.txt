Intrinsic Deformation
=====================

Intrinsic deformations are defined as sequences of speed vectors applied to the points of a shape. The length of such a sequence is computed thanks to a Riemannian metric, whose role is to penalize certain transformation. Choosing such a metric roughly corresponds to penalize some distortions of the shape.


Math
----

In intrinsic deformation is determined by a sequence of velocity vector fields $V = (V^t)_{1 \leq t \leq T}$. The shape after deformation is defined as:

$$\text{Morph}(X_i, V) = X_i + \sum_{t=1}^T V^t_i.$$

We also define intermediate states $(X^m)_{1 \leq m \leq T}$ of the deformation:

$$X_i^{m} = X_i + \sum_{t=1}^m V^t_i$$


The length of the deformation is determined by a Riemannian metric:

$$\text{Length}(X, V^t) = \frac{1}{T}\sum_{t=1}^{T} \ll V^t, V^t \gg_{X^t},$$

where $\ll V^t, V^t \gg_{X^t}$ is a Riemannian metric depending on the position of the points and the topology (edges, triangles) of the shape. This term informs about the "distortion" of the shape caused by the transformation:

$$ X^t \rightarrow X^{t+1} = X^t + V^{t}$$


Code
----

Intrinsic Deformation is accessible in scikit-shapes through the class [`IntrinsicDeformation`](skshapes.morphing.intrinsic_deformation.IntrinsicDeformation). The argument `n_steps` controls the number of time steps $T$, the higher `n_steps` is, the more flexible is the model. However, the memory impact grows linearly in `n_steps` and the running time is also impacted. The Riemannian metric is given with the argument `metric`. Available metrics are:

- [`as-isometric-as-possible`](skshapes.morphing.intrinsic_deformation.as_isometric_as_possible) (requires points and edges)
- [`shell-energy`](skshapes.morphing.intrinsic_deformation.shell_energy_metric) (requires triangles)



```python
import skshapes as sks

loss = ...
metric = sks.ShellEnergyMetric
model = sks.IntrinsicDeformation(n_steps=10, metric=metric)

registration = sks.Registration(loss=loss, model=model)
registration.fit(source=source, target=target)

path = registration.path_
morphed_source = registration.morphed_shape_
```
