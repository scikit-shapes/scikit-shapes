.. _explanation_deformation_rigid_icp:

Rigid ICP (Iterative Closest Point)
===================================

We recall the notations introduced in our :ref:`explanation of rigid deformations <explanation_deformation_rigid>`: a rigid deformation model parameterized by the pair $(R,t)$ transforms a source shape $A = \{a_i\}$ into a target-aligned shape $A_{\theta}$, such that:

.. math::
   \text{Model}(\theta; A) = A_{\theta}, \quad \text{where} \quad \theta = (R, t).

Here, we describe the classic **Rigid ICP** algorithm, designed to estimate the optimal rigid transformation parameters $(R,t)$ aligning the source points $A$ with target points $B = \{b_j\}$.

Parameters
~~~~~~~~~~

- **Rotation matrix** $R \in SO(3)$.
- **Translation vector** $t \in \mathbb{R}^3$.

Weighted ICP Objective
~~~~~~~~~~~~~~~~~~~~~~

Let $\sigma$ represent tentative correspondences, mapping each source point $a_i$ to a matched target point $b_{\sigma(i)}$. Each source point $a_i$ is assigned:

- A positive-definite local metric $L_i \succ 0$, computed from the corresponding point $b_{\sigma(i)}$ encoding how source points can be transformed.
- A scalar weight $w_i > 0$.

Optionally, we include a Gaussian prior on the parameters via a diagonal precision matrix $\Lambda = \lambda I$.

Iteratively Reweighted Least Squares (IRLS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At each IRLS iteration, given the residuals from the previous iteration, we fix scalar weights $\{w_i\}$ (based on a robust estimator). The weighted least-squares cost we minimize is:

.. math::
   C(R, t; \sigma) = \sum_i w_i \lVert R\,a_i + t - b_{\sigma(i)} \rVert_{L_i}^{2} + \lVert \theta \rVert_{\Lambda}^{2},

where:

- $\lVert v \rVert_{L_i}^{2} = v^{\top} \,L_i\, v$ is the Mahalanobis norm squared.
- $\lVert \theta \rVert_{\Lambda}^{2}$ represents the prior regularization term.

Robust M-estimator Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To robustify ICP against outliers, we express our true objective using a robust M-estimator:

.. math::
   C_{\mathrm{M}}(R, t; \sigma) = \sum_{i} \rho(e_i) + \lVert \theta \rVert_{\Lambda}^2,
   \quad
   e_i = \lVert R\,a_i + t - b_{\sigma(i)} \rVert_{L_i},

where $\rho: [0, \infty) \to [0, \infty)$ is a robust loss function (e.g. Huber or Tukey).

At each iteration, we linearize $\rho$ around current residuals $e_i$, defining influence and weights:

.. math::
   \psi(e_i) = \frac{d \,\rho(e_i)}{d\,e_i}, \quad w_i = \frac{\psi(e_i)}{e_i}, \quad (e_i > 0).

If $e_i = 0$, one typically sets $w_i = \psi'(0)$, assuming $\psi'(0)$ is finite.

We solve the weighted least-squares problem above iteratively until convergence, yielding a local minimizer of $C_{\mathrm{M}}$.

Choice of $\rho$ (and corresponding $w_i)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **$L_1$ (absolute value)** – *zero parameter, works “out of the box”*

  .. math::
     \rho(e) = |e|, \qquad
     w(e)   = \frac{1}{|e| + \varepsilon}

* **Variable Trimming (Var-Trim)** – *hard rejection with adaptive overlap*

  .. math::
     \rho(e;\tau) =
       \begin{cases}
         e^2 & \text{if } e \leq \tau,\\
         \tau^2 & \text{otherwise},
       \end{cases}
     \qquad
     w(e;\tau) =
       \begin{cases}
         1 & \text{if } e \leq \tau,\\
         0 & \text{otherwise}.
       \end{cases}

  Here $\tau$ is recomputed at each iteration so that we keep a fixed proportion
  (typically 70–90 %) of the smallest residuals.

* **Cauchy (fixed scale)** – *single break-point $k$*

  .. math::
     \rho(e;k) = \frac{k^2}{2}\,\log\!\bigl(1 + (e/k)^2\bigr), \qquad
     w(e;k)   = \frac{1}{1 + (e/k)^2}

* **Cauchy MAD (auto-scaled)** – *robust, almost parameter-free*

  At each IRLS outer loop compute the median absolute deviation

  .. math::
     s = 1.4826 \,\text{MAD}(e),

  then normalise residuals $\tilde e = e / s$ and reuse the Cauchy form with
  $k \approx 1$:

  .. math::
     \rho(\tilde e) = \frac{k^2}{2}\,\log\!\bigl(1 + (\tilde e/k)^2\bigr), \qquad
     w(\tilde e)   = \frac{1}{1 + (\tilde e/k)^2}

Alternating Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

Rigid ICP alternates between two steps:

1. **Correspondence step:** Identify nearest target points for each source point under the current transformation:

   .. math::
      \sigma(i) = \mathrm{argmin}_j \;\lVert R\,a_i + t - b_j \rVert_{L_i}^2.

   **Compute local metrics** $L_i$ based on the current correspondences $\sigma(i)$.

2. **Registration step:** Solve the weighted Procrustes problem given fixed correspondences:

   .. math::
      (R, t) = \underset{R \in SO(3),\,t \in \mathbb{R}^3}{\mathrm{argmin}}
      \sum_i w_i \lVert R\,a_i + t - b_{\sigma(i)} \rVert_{L_i}^2 + \lVert \theta \rVert_{\Lambda}^2.

   Closed-form solution:

   - Compute weighted centroids:
      .. math::
         \bar{a} = \frac{\sum_i w_i \,a_i}{\sum_i w_i}, \quad
         \bar{b} = \frac{\sum_i w_i \,b_{\sigma(i)}}{\sum_i w_i}.

   - Construct weighted cross-covariance:
      .. math::
         H \;=\; \sum_i w_i\, (a_i - \bar{a})\, (b_{\sigma(i)} - \bar{b})^{\top}.

   - Perform SVD: $H = U \,\Sigma\, V^{\top}$, set $R = V\,U^{\top}$ (correct for $\det R < 0$), then $t = \bar{b} - R\,\bar{a}$.

Iterate until changes in cost $C$ become negligible or a maximum iteration count is reached.

Initialization of $R$ and $t$
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **FPFH + RANSAC:**
  Compute local FPFH descriptor, sample
  minimal sets, and keep the transform that maximises the inlier count.
  *When to use:* medium overlap (≈ 40–80 %), unknown rotation, outliers < 90 %.

- **PCA alignment:**
  Aligns the first (or first two) principal axes of the clouds before refinement.
  *When to use:* strongly elongated or planar geometry (pipes, façades, trunks),
  clouds already fairly close.

- **Centroid + SVD:**
  Solves Procrustes on a handful of coarse matches, then refines.
  *When to use:* high overlap (> 60 %) with some reliable correspondences.

- **Centroid shift only:**
  Simply translates the source cloud onto the target barycentre.
  *When to use:* nearly overlapping captures (multi-frame of the same sensor),
  initial error < 10 cm and < 10°.

Choice of $L_i$
~~~~~~~~~~~~~~~~~

* In many 3D-registration problems, the most reliable constraint at each point $a_i$ is how far it lies from the tangent plane of the underlying surface. When we have normals information, we then use a point-to-plane local metric that is defined as

   .. math::
      L_i = \alpha_i \, n_{\sigma(i)} \, n_{\sigma(i)}^{\top} + \beta (I - n_{\sigma(i)} n_{\sigma(i)}^{\top}),

   where $\alpha\;\gg\; \beta > 0$ are scalar weights, and $n_{\sigma(i)}$ is the normal vector at point $b_{\sigma(i)}$ corresponding to $a_i$.
   With this metric, displacements along $n_{\sigma(i)}$ are penalized more than displacements in the tangent plane.
* In the absence of normal information, a point-to-point (isotropic) metric can be used instead:

   .. math::
      L_i = I,

* If both source and target have normals of good quality, we can use the plane-to-plane metric as an alternative:

   .. math::
      L_i + L'_{\sigma(i)}, \quad
      L_i = \alpha_i\, n_i\, n_i^\top, \quad
      L'_{\sigma(i)} = \alpha'_{\sigma(i)}\, n'_{\sigma(i)}\, {n'_{\sigma(i)}}^\top.

* Finally, for more refined capture of the local geometry of the shapes, one can use the local covariance at the cost of inverting matrices:

   .. math::
      L_i = (\Sigma_i + \Sigma’_{\sigma(i)})^{-1}.

   where $\Sigma_i$ is the local covariance matrix at point $a_i$, and $\Sigma’_{\sigma(i)}$ is the local covariance matrix at point $b_{\sigma(i)}$.
