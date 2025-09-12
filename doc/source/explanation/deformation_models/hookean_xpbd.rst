.. _explanation_deformation_hookean_xpbd:

Hookean elastic material with XPBD
========================================

This document combines the :ref:`XPBD formulation <explanation_deformation_xpbd>` with the :ref:`Hookean energy model <explanation_deformation_hookean>` to derive the Hookean-elastic constraint and its XPBD compliance.

Constraint function and compliance for the Hookean model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We know that the elastic energy potential derived from a single constraint $\mathcal C_j$ in the XPBD model is

.. math::

    U(x)=\frac{1}{2}\alpha_j^{-1}\left(\mathcal{C}_j(x)\right)^2,

where $\alpha_j$ is the compliance factor corresponding to the constraint.

On the other hand, the total elastic strain energy for an element of volume :math:`V_e` stored in a Hookean material due to deformation is given by:

.. math::

    W_{\mathrm{tot}}(\mathbf{F}) = V_e\,W(\mathbf{F}) = V_e\,E\,\widehat{W}(\mathbf{F})

where :math:`W(\mathbf{F})` is the elastic strain energy density and :math:`\widehat{W}(\mathbf{F})` is the elastic strain energy density, normalized by Young's modulus :math:`E`, defined by:

  .. math::

      W_{\mathrm{tot}}(\mathbf{F}) = Ve\,W(\mathbf{F})\quad\text{and}\quad W(\mathbf{F}) = E\,\widehat{W}(\mathbf{F}).

When simulating an Hookean elastic material with XPBD, both expressions of the elastic enargy equalize

.. math::

    \frac{1}{2}\alpha_j^{-1}\left(\mathcal{C}_j(\mathbf{F})\right)^2 = V_e E \,\widehat{W}(\mathbf{F})

so $\alpha_j$ is proportional to $\frac{1}{V_e E}$, and we have

.. math::

    C_{\mathrm{Hooke}}(\mathbf{F}) = \sqrt{2\,\widehat{W}(\mathbf{F})}.

This constraint function evaluates to zero when the material is in its undeformed state (:math:`\mathbf{F} = \mathbf{I}_3`), and it evaluates to a positive value whenever deformation occurs (:math:`\mathbf{F} \neq \mathbf{I}_3`).

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
