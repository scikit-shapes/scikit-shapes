.. _explanation_deformation_rigid_icp:

Rigid ICP (Iterative Closest Point)
===================================

We recall the notations of our :ref:`explanation of rigid deformations<explanation_deformation_rigid>`:
$\theta$ denotes the couple $(R,t)$ of parameters of a rigid deformation model
that maps a source shape $A=\{a_i\}$ to a target shape $B=\{b_i\}$:
$\text{Model}(\,\theta\,;\,A\,)=A_{\theta}$
This page describes the classic **Rigid ICP** algorithm, which estimates the parameters $(R,t)$ of the
rigid transformation that maps $A$ to $A_{\theta}$.

Parameters
~~~~~~~~~~~
The rigid transformation is parameterised exactly as in the rigid deformation model:

- Rotation matrix $R \in SO(3)$.
- Translation vector $t \in \mathbb{R}^3$.

Rigid ICP cost
~~~~~~~~~~~~~~~
We define the full ICP objective as a squared residuals function of $(R,t)$ and the correspondence $\sigma$:

.. math::
   C\bigl(R,t;\sigma\bigr)
   \;=\;
   \sum_i \bigl\|R\,a_i + t - b_{\sigma(i)}\bigr\|^2.

where $\{a_i\}$ is the source shape, $\{b_i\}$ is the target shape, and $\sigma(i)$ is the index of the closest point in $\{b_i\}$ to $a_i$.
In practice we never optimize jointly over $(R,t)$ and $\sigma$ who is combinatorial. Instead we alternate the two following steps:
	1.	Fix $(R,t)$, update $\sigma$.
	2.	Fix $\sigma$, update $(R,t)$.

Alternating optimization scheme
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Rigid ICP alternates between two main steps until convergence:

#. **Correspondence step**
   Given the current estimate $(R,t)$, find for each source point $a_i\in A$ its closest target
   point $b_{\sigma(i)}\in A_{\theta}$:

   .. math::
      \sigma(i) = \text{argmin}_j\vert R a_i + t - b_j\vert^2.

#. **Registration step**
   Compute the optimal $(R,t)$ aligning matched pairs
   $\{(a_i,b_{\sigma(i)})\}$ by solving:

   .. math::
      (R,t) = \underset{R\in SO(3),\,t\in\mathbb R^3}{\text{argmin}}\;
      \sum_i \left\vert R a_i + t - b_{\sigma(i)}\right\vert^2,

   typically via SVD of the cross-covariance matrix.

These steps repeat until convergence.

Initialization
~~~~~~~~~~~~~~~
A good initial guess $(R_0,t_0)$ is crucial to avoid poor local minima. Common strategies include:

- **Orthogonal Procrustes**
  When a rough correspondence or subset of matches $\{(a_i,b_i)\}$ is available,
  1. Compute centroids:

     .. math::
        \mu_A = \frac{1}{n}\sum_i a_i,\quad
        \mu_B = \frac{1}{n}\sum_i b_i.

  2. Center points:

     .. math::
        \widetilde{a}_i = a_i - \mu_A,\quad
        \widetilde{b}_i = b_i - \mu_B.

  3. Compute cross-covariance:

     .. math::
        H = \sum_i \widetilde{a}_i \widetilde{b}_i^\top.

  4. Perform SVD and extract rotation and translation:

     .. math::
        H = U \Sigma V^\top,\quad
        R_0 = V U^\top,\quad
        t_0 = \mu_B - R_0\mu_A.

- **Centroid alignment only**
  Set $R_0 = I$, $t_0 = \mu_B - \mu_A$ if no correspondences are known.
