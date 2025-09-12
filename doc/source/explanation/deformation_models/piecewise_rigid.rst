.. _explanation_deformation_piecewise_rigid:

Piece-wise rigid transformations
================================

We recall the notations of our :ref:`introduction to shape registration<explanation_registration>`:
$\theta$ denotes the vector of P parameters of a deformation model
that maps a source shape $A$ to a deformed shape
$\text{Model}(\,\theta\,;\,A\,)=A_{\theta}$.
Any such deformation is associated to a regularization penalty $\text{Regularization}(\theta)$.

This page describes "piece-wise rigid" deformation models. :ref:`Rigid deformation models<explanation_deformation_rigid>` are sometimes
too restrictive for shape deformations where sub-parts can move independently but each sub-part
deformation is a rigid deformation (e.g. robotic hands, video game characters, ...).
Piece-wise rigid models apply a separate rigid transformation to each sub-part of the shape.
In these models, the main property of rigid transformations - that they preserve the Euclidean distance between
every pair of points - is still satisfied, piece-wise.

Parameters
~~~~~~~~~~
Let's say the source shape has $K$ sub-parts (bones, rigid blocks, ...). We can write it as

.. math::

   A = \bigcup_{k=1}^K A_k

where $A_k$ is the $k^{th}$ sub-part (or piece) of the shape.

Each piece $A_k$ has its own rigid parameters $(R_k, t_k)\in SO(3)\times \mathbb R^3$.
The complete parameter vector is

.. math::

   \theta = \{(R_k,\,t_k)\}_{k=1}^{K}

.. note::
   We require that the rigid transformations are proper, i.e. $\det R_k = +1$, in order to keep orientation consistency and physical plausibility.

Total deformation
~~~~~~~~~~~~~~~~~
If $X\sim A$ and $X_\theta\sim{}A_\theta$ denote the **vectors of features** of length F
that represent the source and deformed shapes, and if $X^{(k)}\sim A_k$ and
$X_\theta^{(k)}\sim{}A_k^\theta$ denote the vectors of features of the source and deformed $k^{th}$ sub-part,
then the deformation of the $k^{th}$ sub-part is given by

.. math::

   X_\theta^{(k)} = R_k\,X^{(k)} + t_k.

The induced deformation for the whole shape is then

.. math::

   X_\theta =\bigcup_{k=1}^{K}\bigl(R_k\,X^{(k)} + t_k\bigr).

What happens at the intersection of two pieces?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We've seen that the deformation of each piece is a rigid transformation.
If the pieces are disjoint, the deformation is well defined. This is the usual setting, where each vertex belongs to a single piece.
But if the pieces overlap, we need to define how to handle the deformation of the overlapping region.
The most common solution is to use a weighted average of the deformations of the overlapping subparts,
as in Linear Blend Skinning.

If $p$ is a point in the overlapping region of the source shape, we can write its deformation as

.. math::

   p_\theta = \sum_k w_k(p)\bigl(R_k\,p + t_k\bigr),

where the weight field (that should be smooth in most cases) sums to $1$ on $k$ : $\sum_k w_k(p) = 1$.

This ensures a smooth transition between the overlapping deformations, but at the cost of breaking strict rigidity in the overlapping regions.
Indeed, the weighted average of rigid transformations is not, in general, a rigid transformation. This introduces local non-rigid behavior,
which can lead to the well-known "candy wrapper" artifacts.

Dual quaternion skinning solves this problem when there is only two overlapping pieces, but still presents bulging artifacts
when there are more than two overlapping pieces.


Regularization
~~~~~~~~~~~~~~
Unlike pure rigid deformations, piece-wise rigid deformations often require a regularization term to avoid
overfitting and guarantee that the deformation is plausible in the context of the application.

Usually, a simple stiffness regularization is used, in addition to a translation penalization term:

.. math::

   \text{Regularization}(\theta) = \sum_{(k,\ell)\in\mathcal{N}} \Vert R_k-R_\ell\Vert^2_F + \lambda\sum_k\Vert t_k\Vert^2

where $\mathcal{N}$ is the set of neighbouring pieces, $\|\cdot\|_F$ is the Frobenius norm and $\lambda$
is a hyperparameter that controls the strength of the regularization.
