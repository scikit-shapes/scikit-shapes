Extrinsic Deformation
=====================

Presentation
------------

Extrinsic deformation should be understood as deformation of the ambient space, the shape deformation follows.

The points on which the deformation is defined could be the points of the shape itself (`control_points=False`) or another set of points (`control_points=True`). Usual choice of control points are vertices of regular grid around the shape.

The parameter is a vector field $p$ of momenta. These momenta have to be thought as intention for each point of $q$ to go in a certain direction. The actual velocity is computed through local averaging of momenta using a kernel operator.

Under the umbrella of extrinsic deformation falls classical deformation models such as splines or LDDMM (Large deformation diffeomorphic metric mapping)


Math
----

- $X = (x_i)_{1\leq i\leq n}$ : points of the shape
- $C = (c_i)_{1\leq i\leq nc}$ : control points
- $P = (p_i)_{1\leq i\leq nc}$ : momentum
- $K$: a kernel operator

If `n_steps = 1`:

$$ \text{Morph}(X) = X + K_{X}^C P. $$

If `n_steps > 1`:

Let us consider the hamiltonian: $H(P, Q) = <P, K_Q^Q P> / 2$. The following differential equation serves to define the transformation

- $P(t = 0) = P$, $Q(t = 0) = C$, $X(t = 0) = X$
- $\dot{P} = - \frac{\partial}{\partial Q} H(P, Q)$
- $\dot{Q} = \frac{\partial}{\partial P} H(P, Q) = K_Q^Q P$
- $\dot{X} = K_Q^X P$

The transformed shape is $X(t = 1)$.


The length of the deformation is given by :

$$<P, K_Q^Q P> / 2$$



Code
----

Extrinsic Deformation is accessible in scikit-shapes through the class [`ExtrinsicDeformation`](skshapes.morphing.extrinsic_deformation.ExtrinsicDeformation).


```python
import skshapes as sks

loss = ...
model = sks.ExtrinsicDeformation(n_steps=1, kernel="gaussian", scale=0.1)

registration = sks.Registration(loss=loss, model=model)
registration.fit(source=source, target=target)

path = registration.path_
morphed_source = registration.morphed_shape_
```

Examples
--------

TBA
