Extrinsic Deformation
=====================

Presentation
------------

Extrinsic deformations are defined with a vector field of momentum, that can be either defined on the points of the shape or on control points.

Math
----

- $X = (x_i)_{1\leq i\leq n}$ : points of the shape
- $C = (c_i)_{1\leq i\leq nc}$ : control points
- $P = (p_i)_{1\leq i\leq nc}$ : momentum
- $K$: a kernel operator

If `n_steps = 1`:

$$ \text{Morph}(X) = X + K_{X}^C P. $$

If `n_steps > 1`:

Let us consider the hamiltonian: $H(P, Q) = <P, K_Q^Q P> / 2$

- $P(t = 0) = P$, $Q(t = 0) = C$, $X(t = 0) = X$
- $\dot{P} = - \frac{\partial}{\partial Q} H(P, Q)$
- $\dot{Q} = \frac{\partial}{\partial P} H(P, Q) = K_Q^Q P$
- $\dot{X} = K_Q^X P$

The transformed shape is $X(t = 1)$.


The length of the deformation is determined by :

$$<P, K_Q^Q P> / 2$$



Code
----

Extrinsic Deformation is accessible in scikit-shapes through the class [`ExtrinsicDeformation`](skshapes.morphing.extrinsic_deformation.ExtrinsicDeformation).


```python
import skshapes as sks

loss = ...
model = sks.ExtrinsicDeformation(n_steps=1, kernel=sks.GaussianKernel(0.1))

registration = sks.Registration(loss=loss, model=model)
registration.fit(source=source, target=target)

path = registration.path_
morphed_source = registration.morphed_shape_
```
