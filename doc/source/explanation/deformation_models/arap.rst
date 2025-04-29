.. _explanation_deformation_arap:

As-Rigid-As-Possible (ARAP) Deformations
=========================================

This page describes **As-Rigid-As-Possible (ARAP)** deformation models.
Piece-wise rigid models allow independent rigid transformations for each part,
which can introduce discontinuities or blending artifacts at the transitions between parts.
ARAP models relax piece-wise rigidity by seeking deformations that are locally as rigid as possible,
leading to more globally coherent results. The intuition is to use a discretized version of the Laplace-Beltrami
operator to measure the deviation from being locally rigid.
Note that ARAP works well for meshes, and can also be applied to point clouds via neighbor graphs.

ARAP energy
~~~~~~~~~~~~~~~~~~~~~~~
Given source vertex positions $\mathbf{p}=\{p_i\}$ and a set of positional constraints (handles)

.. math::
   p'_k = c_k, \qquad k\in\mathcal{F},

ARAP model finds new vertex positions $\mathbf{p}'=\{p'_i\}$ that satisfy the constraints and minimize the local deviation from rigid deformations. This can be measured by the non-convex ARAP energy

.. math::
   E(\mathbf{p}',\{R_i\})
   =\sum_i w_i \sum_{j\in\mathcal{N}(i)}
   w_{ij}\left\Vert(p'_i-p'_j)-R_i(p_i-p_j)\right\Vert^2,

where

- the per-vertex rotations $R_i \in SO(3)$ act as latent variables: they are not prescribed in advance, but are instead optimized internally during the alternating minimization process.
- $\mathcal{N}(i)$ denotes the neighbors of vertex $i$.
- $w_{ij}$ are scalar weights, we want them to compensate for non-uniformly shaped cells and prevent discretization bias. For meshes, we typically take cotangent weights because they encode local geometry very well:

   .. math::

      w_{ij} = \frac{1}{2}(\cot \alpha_{ij} + \cot \beta_{ij}),

   where $\alpha_{ij}$ and $\beta_{ij}$ are the angles opposite to the edge $(i,j)$ in the mesh triangles.
   Furthermore, for these weights, the weighted sum $\sum_{j\in\mathcal{N}(i)} w_{ij} (p'_i - p'_j)$ discretizes
   the coordinate-wise Laplace–Beltrami operator applied to the position function $\mathbf{p}'$.
   In flat Euclidean spaces, this operator vanishes under rigid deformations, whereas non-affine deformations produce a non-zero
   Laplacian.
   For meshless ARAP (with a neighbor graph), we can use constant or distance-based weights (e.g. $w_{ij} = 1/\left\Vert p_i - p_j \right\Vert^2$) instead of cotangent weights.

The ARAP energy can be interpreted as a generalized mass-spring system: each edge $(i,j)$ behaves like a spring
whose rest length and orientation are dynamically adjusted via the local rotation $R_i$. Instead of preserving only
distances (as in a classical spring system), ARAP springs attempt to preserve the local rigid motion (both distances and relative orientations).
The weights $w_{ij}$ act as stiffnesses for the springs, encoding the local geometric structure. Unlike classical springs, ARAP springs adjust both the rest
length and local frame to minimize deformation.
$SO(3)$ being non-convex, the minimization of the ARAP energy is a non-convex problem, and the solution may converge to a local minimum.
Since both the vertex positions $\mathbf{p}'$ and the local rotations $R_i$ are unknown, we use an
alternating minimization approach: at each iteration, we fix one set of variables and optimize over the other, in two steps.

Alternating minimization approach
~~~~~~~~~~~~~~~~~~~~~~~

1. **Local step** – closed-form update of $R_i$.
   For fixed $\mathbf{p}'$, assemble the covariance

   .. math::
      S_i=\sum_{j\in\mathcal{N}(i)}
           w_{ij}(p_i-p_j)(p'_i-p'_j)^{\top},

   then perform SVD (singular value decomposition) $S_i=U_i\Sigma_iV_i^{\top}$ to extract the optimal rotation aligning the original and deformed edges at $i$

   .. math::
      R_i = V_iU_i^{\top}.

   We implement the following details to ensure robustness:
   - If $\det(R_i) < 0$ after the SVD decomposition $S_i = U_i \Sigma_i V_i^\top$, flip the sign of the last column
   of $V_i$ before recomputing $R_i = V_i U_i^\top$, to ensure a proper rotation ($\det(R_i) > 0$).
   - When $S_i$ is near-singular (for example, if neighbors are nearly colinear or coplanar),
   small numerical instabilities may occur during SVD. To improve robustness, it can help to regularize the
   covariance by adding a small multiple of the identity matrix, $S_i \leftarrow S_i + \epsilon I$, with a tiny
   $\epsilon > 0$.

2. **Global step** – solve a sparse SPD (symmetric positive definite, thus uniquely solvable) system for $\mathbf{p}'$.
   $\frac{\partial E}{\partial p'_i}=0$ gives the linear sparse system

   .. math::
      \sum_{j\in\mathcal{N}(i)} w_{ij}(p'_i-p'_j)
      =\sum_{j\in\mathcal{N}(i)}
        \frac{w_{ij}}{2}\left(R_i\!+\!R_j\right)(p_i-p_j).

   which rewrites in matrix form as $\mathbf{L}\mathbf{p}'=\mathbf{b}$ with
   $\mathbf{L}$ the SPD cotangent-weighted discrete Laplacian. We apply Dirichlet boundary conditions
   for the constrained indices $\mathcal{F}$ by erasing the
   corresponding rows and columns from the Laplacian matrix $\mathbf{L}$,
   and updating the right-hand side $\mathbf{b}$ accordingly to substitute
   the fixed values $p'_k = c_k$. Erasing rows and columns to apply Dirichlet constraints
   is equivalent to modifying the original system by assigning infinite stiffness to the
   constrained vertices. We then find the unique solution guaranteed by symmetric positive definiteness
   using precomputed Cholesky factorization of $\mathbf{L}$.

**Remarks:**

- The local steps at each vertex $i$ are independent and can be parallelized across vertices for efficiency.
- Each iteration (local step + global step) monotonically decreases the ARAP energy, typically leading to fast convergence to a local minimum in practice. In practice, for small deformations and good initializations, a few iterations are sufficient to converge to a good minimum.

References
~~~~~~~~~~

.. [SorkineAlexa07] Olga Sorkine and Marc Alexa. "As-Rigid-As-Possible Surface Modeling."
   In Symposium on Geometry processing (Vol. 4, pp. 109-116), 2007.
