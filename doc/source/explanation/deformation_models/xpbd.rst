.. _explanation_deformation_xpbd:

Extended Position-Based Dynamics (XPBD)
========================================

We inherit from :ref:`multi-affine models <explanation_deformation_multi_affine>`.

Extended Position-Based Dynamics (XPBD) is an optimization method designed to iteratively enforce constraints by applying geometric updates directly on point positions. It is particularly suited for simulations requiring stable and realistic physical deformations, and is very robust to long time steps and low iteration counts.

XPBD extends classical Position-Based Dynamics (PBD) by introducing a compliance parameter :math:`\alpha_j \geq 0`, allowing constraints to behave elastically.

Classical Position-Based Dynamics (PBD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~

XPBD extends the classical Position-Based Dynamics by incorporating a compliance parameter :math:`\alpha_j \geq 0` for each constraint, introducing elasticity into constraints. Constraints can thus behave elastically, allowing minor violations to improve numerical stability and physical realism:

.. math::

    \mathcal{C_j}(\mathbf{x^{t+1}}) = -\tilde\alpha_j \lambda_j^{t+1}

where :math:`\lambda^{t+1}` is a Lagrange multiplier at iteration :math:`t+1`. Note that we use here :math:`\tilde\alpha_j = \alpha_j / \Delta t^2` to ensure that the compliance parameter is independent of the time step.

The compliance parameter is also directly correlated to the elastic energy potential derived from a constraint :math:`\mathcal{C}` by:

.. math::
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
