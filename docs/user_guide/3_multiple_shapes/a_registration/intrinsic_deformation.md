## Presentation



## Math

In intrinsic deformation is determined by a sequence of velocity vector fields $V = (V^t)_{1 \leq t \leq T}$. The shape after deformation is defined as

$$\text{Morph}(X_i, V) = X_i + \sum_{t=1}^T V^t_i$$.

We also defined intermediate states of the deformation

$$X_i^{m} = X_i + \sum_{t=1}^m V^t_i$$


The length of the deformation is determined by a Riemannian metric

$$\text{length}(X, V^t)$ = \sum \ll V^t, V^t \gg_{X^t} $$


## Code

Rigid motion is accessible in scikit-shapes through the class [`IntrinsicDeformation`][skshapes.morphing.IntrinsicDeformation]. The only argument is `n_steps`. Note that this argument has no influence at all on the optimization step, its only influence is if you want to create an animation and have intermediate steps

```python
import skshapes as sks

loss = ...
model = sks.RigidMotion()

registration = sks.Registration(loss=loss, model=model)
registration.fit(source=source, target=target)

path = registration.path_
morphed_source = registration.morphed_shape_
```