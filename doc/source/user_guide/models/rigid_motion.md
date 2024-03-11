Rigid Deformation
=================

Rigid motion corresponds corresponds to a change of shape position and/or orientation while the size and the form of the shape stays the same. Using rigid motion for registration is particularly useful for aligning data.


Math
----

A rigid motion is determined by a rotation matrix $R$ and a translation vector $t$. If $X = (X_i)_{0\leq i\leq N}$ denotes the $N \times d$ array of vertices the transformation can be written as

$$\text{Morph}(X_i) = \bar{X} + R (X_i - \bar{X}) + t$$

- In 2D, the rotation is determined by an angle $\theta$ and the rotation matrix is $R = \begin{pmatrix} \cos \theta & - \sin \theta \\ \sin \theta & \cos \theta \end{pmatrix}$
- In 3D, the rotation is represented by three Euler angles $(\psi, \theta, \phi)$, representing a sequence of rotations around axis. See the [Wikipedia entry](https://en.wikipedia.org/wiki/Euler_angles) to get more information about Euler angles.

Code
----

Rigid motion is accessible in scikit-shapes through the class [`RigidMotion`](skshapes.morphing.rigid_motion.RigidMotion). The only argument is `n_steps`. Note that this argument has no influence at all on the optimization step, its only influence is if you want to create an animation and have intermediate steps

```python
import skshapes as sks

loss = ...
model = sks.RigidMotion()

registration = sks.Registration(loss=loss, model=model)
registration.fit(source=source, target=target)

path = registration.path_
morphed_source = registration.morphed_shape_
```
