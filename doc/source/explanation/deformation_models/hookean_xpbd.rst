.. _explanation_deformation_hookean_xpbd:

Hookean elastic material with XPBD
========================================

We inherit from :ref:`multi-affine models <explanation_deformation_multi_affine>`.

Extended Position-Based Dynamics (XPBD) is an optimization method designed to iteratively enforce constraints by applying geometric updates directly on point positions. It is particularly suited for simulations requiring stable and realistic physical deformations, and is very robust to long time steps and low iteration counts.

Mathematical Background
~~~~~~~~~~~~~~~~~~~~~~~

XPBD extends classical Position-Based Dynamics (PBD) by incorporating a compliance term to constraints, introducing elasticity into constraints. It has a direct correspondence to well-known energy potential and allows to solve constraints in a time step and iteration count independent manner.

Classical Position-Based Dynamics (PBD)
---------------------------------------

Mathematically, classical PBD solves constrained optimization problems through iterative projections of the current solution onto constraint manifolds defined by energy potentials:

.. math::

    \mathbf{x}^{t+1} \gets \mathbf{x}^{t} + \Delta \mathbf{x}, \quad\text{s.t.}\quad \mathcal{C}(\mathbf{x}^{t+1}) = 0

where:

- :math:`\mathbf{x}^{t}` is the position of the particles at iteration :math:`t`.
- :math:`\mathcal{C}` represents the vector of constraint functions.
- :math:`\Delta \mathbf{x}` is the position correction, ensuring constraint satisfaction.

In classical PBD, each iteration applies a constraint correction step computed as:

.. math::

    \Delta \mathbf{x} = k_j s_j \mathbf{M}^{-1}\nabla \mathcal{C}_j(\mathbf{x}^{t})

with the scaling factor :math:`s_j` calculated through a single Newton iteration step:

.. math::

    s_j = \frac{-\mathcal{C}_j(\mathbf{x}^{t})}
    {\nabla \mathcal{C}_j(\mathbf{x}^{t}) \mathbf{M}^{-1}\nabla \mathcal{C}_j(\mathbf{x}^{t})^T}

where:

- :math:`j` indexes the constraints.
- :math:`k_j \in [0, 1]` is the constraint stiffness.
- :math:`\mathbf{M}` is the diagonal mass matrix.
- :math:`\nabla \mathcal{C}_j(\mathbf{x}^{t})` denotes the gradient of the constraint :math:`\mathcal{C}_j`.

XPBD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

XPBD extends the classical Position-Based Dynamics by incorporating a compliance parameter :math:`\alpha_j \geq 0` for each constraint, introducing elasticity into constraints. Constraints can thus behave elastically, allowing minor violations to improve numerical stability and physical realism:

.. math::

    \mathcal{C_j}(\mathbf{x^{t+1}}) = -\tilde\alpha_j \lambda_j^{t+1}

where :math:`\lambda^{t+1}` is a Lagrange multiplier at iteration :math:`t+1`. Note that we use here :math:`\tilde\alpha_j = \alpha_j / \Delta t^2` to ensure that the compliance parameter is independent of the time step.

The compliance parameter is also directly correlated to the elastic energy potential derived from a constraint :math:`\mathcal{C}` by:

.. math::
    :label: potential_energy_xpbd
    :nowrap:

    \begin{align}
    U(\mathbf{x}) = \frac{1}{2}\mathcal{C}(\mathbf{x})^T \alpha^{-1}\mathcal{C}(\mathbf{x})
    \end{align}

where :math:`\alpha` is the block diagonal compliance matrix representing inverse constraint stiffness.

XPBD iteratively solves the constraint system using incremental updates of the Lagrange multipliers :math:`\lambda_j`. At each iteration :math:`i`, the incremental Lagrange multiplier :math:`\Delta \lambda_j` for a constraint :math:`j` is computed by solving the regularized linear equation derived from a Gauss-Seidel style update step:

.. math::

    \Delta \lambda_j = \frac{-\mathcal{C}_j(\mathbf{x}^{t}) - \tilde\alpha_j \lambda^t_{j}}
    {\nabla \mathcal{C}_j(\mathbf{x}^{t}) \mathbf{M}^{-1}\nabla \mathcal{C}_j(\mathbf{x}^{t})^T + \tilde\alpha_j}

- :math:`\lambda^t_{j}` represents the accumulated Lagrange multiplier at iteration :math:`t` for constraint :math:`j`.
- When :math:`\alpha_j = 0`, XPBD reduces exactly to classical PBD.
- For :math:`\alpha_j > 0`, an additional regularization term is included in both the numerator and denominator, limiting the magnitude of the constraint force.

The corresponding position update is computed as:

.. math::

    \Delta \mathbf{x} = \Delta \lambda_j \mathbf{M}^{-1}\nabla \mathcal{C}_j(\mathbf{x}^{t}).

Hookean elastic model
---------------------

The **Hookean elastic model** describes linear elastic behavior of materials through a direct proportionality between stress and strain. Hooke's law states:

.. math::

    \mathbf{S} = \mathbf{C}\,\boldsymbol{\varepsilon}

where:

- :math:`\mathbf{S}` is the stress tensor (force per unit area).
- :math:`\mathbf{C}` is the fourth-order elasticity tensor encoding material stiffness.
- :math:`\boldsymbol{\varepsilon}` is the Green-Lagrange strain tensor defined as:

.. math::

    \boldsymbol{\varepsilon} = \frac{1}{2}(\mathbf{F}^T \mathbf{F} - \mathbf{I})

where :math:`\mathbf{F}` is the deformation gradient and :math:`\mathbf{I}` is the identity matrix.

The Hookean model corresponds to a quadratic energy potential given by:

.. math::
    :label: potential_energy_hookean
    :nowrap:

    \begin{align}
    W(\mathbf{F}) = \frac{1}{2}\boldsymbol{\varepsilon}:\mathbf{C}\boldsymbol{\varepsilon}
    \end{align}

where ":" denotes the double inner product, or tensor contraction.

Constraint function and compliance for the Hookean model
--------------------------------------------------------

Using :eq:`eq:potential_energy_xpbd`, we obtain the constraint function for the Hookean model:

.. math::

    C_{\mathrm{Hooke}}(\mathbf{F}) = \sqrt{2\,\widehat{W}(\mathbf{F})}

where :math:`\widehat{W}(\mathbf{F})` is the elastic energy density, normalized by Young's modulus :math:`E`, defined by:

  .. math::

      W(\mathbf{F}) = E\,\widehat{W}(\mathbf{F})

Here, :math:`W(\mathbf{F})` represents the elastic strain energy stored in the material due to deformation.

This constraint function evaluates to zero when the material is in its undeformed state (:math:`\mathbf{F} = \mathbf{I}_3`), and it evaluates to a positive value whenever deformation occurs (:math:`\mathbf{F} \neq \mathbf{I}_3`).

In the Hookean Model, the material's stiffness is measured by the Young's modulus :math:`E`. The total elastic energy for an element of volume :math:`V_e` is given by:

.. math::

    W_{\mathrm{tot}}(\mathbf{F}) = V_e\,W(\mathbf{F}) = V_e\,E\,\widehat{W}(\mathbf{F})

Or, in the XPBD framework, the stiffness of an element of volume :math:`V_e` is given by :math:`\alpha^{-1}`. This means that the compliance parameter :math:`\alpha` for the Hookean model is directly proportional to :math:`\frac{1}{Ve\,E}`.

Constraint gradient
-------------------

The gradient of the Hookean constraint with respect to the position :math:`\mathbf{x}_i` of each particle is derived using the chain rule as:

.. math::

    \nabla_{\mathbf{x}_i}C_{\mathrm{Hooke}}(\mathbf{x})
    =
    \frac{1}{C_{\mathrm{Hooke}}(\mathbf{x})}\nabla_{\mathbf{x}_i}\widehat{W}(\mathbf{F}).

The derivative of :math:`\widehat{W}(\mathbf{F})` with respect to :math:`\mathbf{F}` can be explicitly computed as:

.. math::

    \frac{\partial \widehat{W}(\mathbf{F})}{\partial \mathbf{F}} = \mathbf{S}\mathbf{F}

The deformation gradient :math:`\mathbf{F}` is typically computed as:

.. math::

    \mathbf{F} = \left(\sum_i m_i \mathbf{r}_i\bar{\mathbf{r}}_i^T\right)\,Q^{-1},

where:

- :math:`\mathbf{r}_i=x_i-x_{cm}` is the position of particle :math:`i` relative to center of mass.
- :math:`\bar{\mathbf{r}}_i=\bar{x}_i-\bar{x}_{cm}` is the rest position of particle :math:`i` relative to rest center of mass.
- :math:`Q = \left(\sum_i m_i \bar{\mathbf{r}}_i\bar{\mathbf{r}}_i^T\right)` is the rest-state inertia tensor, which encodes the distribution of mass and initial configuration of the particles.

When differentiating :math:`\mathbf{F}` with respect to :math:`\mathbf{x}_i`, we obtain a direct relation involving the matrix :math:`\mathbf{Q}` and the rest position vectors :math:`\bar{\mathbf{r}}_i`:

.. math::

    \frac{\partial \mathbf{F}}{\partial \mathbf{x}_i} = m_i\,\mathbf{Q}^{-T}\bar{\mathbf{r}}_i

Substituting these results into our chain rule expression, we obtain a clear, compact expression of the gradient for the Hookean constraint:

.. math::

    \nabla_{\mathbf{x}_i}C_{\mathrm{Hooke}}(\mathbf{x})
    =
    \frac{m_i}{C_{\mathrm{Hooke}}(\mathbf{x})}\mathbf{S}\mathbf{F}\mathbf{Q}^{-T}\bar{\mathbf{r}}_i.
