.. _explanation_deformation_rigid:

Rigid transformations (translation + rotation)
==============================================

We recall the notations of our :ref:`introduction to shape registration<explanation_registration>`:
$\theta$ denotes the vector of $P$ parameters of a deformation model
that maps a source shape $A$ to a deformed shape
$\text{Model}(\theta;A)=A_{\theta}$.
Any such deformation is associated to a regularisation penalty
$\text{Regularization}(\theta)$.

This page describes "rigid" deformation models that satisfy the following property:

1. **Rigidity** -- the model preserves the Euclidean distance between every pair of points.
   Then the deformation is either a rotation followed by a translation, either a reflection
   followed by a translation, and there exists an orthogonal matrix $R$ and a translation
   vector $t$ such that the deformed shape $X_\theta$ is given from the source shape $X$ by:

   .. math::

      X_{\theta} \;=\; R\,X + t.

We make an important distinction between rigid transformations that
preserve handedness/chirality (proper rigid transformations or
rigid motions) and those that do not (improper rigid transformations).
Improper rigid transformations flip the orientation of the shape, which
must be corrected later on, many geometric features (normals, mean
curvature, PFH, ...) relying on a consistent orientation.

* If $\det R = +1$ (rotation), the transformation is said to be *proper*.
* If $\det R = -1$ (reflection), the transformation involves a reflection and is said to be
  *improper*.

.. list-table::
   :widths: 25 25
   :header-rows: 1

   * - Rigid transformation
     - Preserves handedness ?
   * - Translation
     - Yes
   * - Rotation
     - Yes
   * - Reflection
     - No

Parameters
~~~~~~~~~~

The parameters of a rigid transformation are decomposed in the rotation parameters and the translation parameters.

- **Rotation parameter** -- we accept five equivalent representations for the rotation parameter:

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Name
     - Symbol
     - Trade-offs
   * - Rotation matrix
     - $R \in O(\mathbb R^3)$
     - 9 parameters with orthogonality constraints, no singularities
   * - Rotation vector (axis-angle)
     - $\omega \in \mathbb R^3$
     - Minimal representation (3 parameters), intuitive axis-angle, singularity at rotation angle $\pi$
   * - Unit quaternion
     - $q=(w,x,y,z)\in\mathbb S^3$
     - Robust interpolation, requires normalization, double-cover $q$ and $-q$

- **Translation parameter** -- it is always given by a vector $t\in\mathbb R^3$.

.. note::

   Unless `allow_reflection=True`, passing an improper rotation
   ($\det R<0$) raises ``ValueError``.

Regularization
~~~~~~~~~~~~~~~~

Rigid transformations preserve the geometry of the shape, so no geometric
regularisation is needed.  Nevertheless, optimisation often benefits
from a prior on the registration parameters to avoid drift and to encode prior knowledge
(e.g. small rotations around an initial guess).  We therefore allow, but do not require, a
quadratic penalty of the form

.. math::

  \text{Regularization}(\theta) ;=; |\theta|_{\Lambda}^{2},

where $\Lambda$ is a symmetric positive-definite matrix.  In the simplest case
$\Lambda = \lambda I$ (ridge penalty) with a small $\lambda > 0$.
