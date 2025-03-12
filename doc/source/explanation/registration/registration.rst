.. _explanation_registration:

Overview of the registration pipeline
=====================================


Let us consider a source shape $A$ and a target shape $B$.

We formulate the registration of $A$ onto $B$ as an optimization problem
of the form:


.. math::

    \theta^\star ~\gets~ \arg \min_{\theta} ~&
    \underbrace{
    \text{Regularization}(\theta) ~+~ \text{Loss}(\,\text{Model}(\,\theta \,;\, A\,),~ B\,)
    }_{\text{Objective}(\theta)}
    \\
    &\text{subject to}~~~\text{Constraints}(\theta) = E~.

In the equation above:

  - $\theta$ is a **vector of parameters** of length P.
    It characterizes the transformation $A_\theta = \text{Model}(\,\theta \,;\, A\,)$
    from the source shape $A$ onto the target shape $B$.
    For instance, it could be a (flattened) **transformation matrix**
    for an affine registration, or a set of **control points** for a free-form deformation.

  - $\text{Regularization}(\theta)$ is a non-negative scalar that penalizes
    the deviation of the parameter vector $\theta$ from a set of "desirable"
    or "plausible" parameters.
    It is often used to enforce **smoothness** or **biomechanical constraints**.

  - :math:`\text{Loss}(A_\theta,B)` is a non-negative scalar that measures the discrepancy
    between the transformed source shape $A_\theta$ and the target shape $B$.
    Common examples include the **closest point** ("Hausdorff") or
    **optimal transport** ("Wasserstein") distances.

  - $\text{Constraints}(\theta) = E$ is a set of equality constraints that must be satisfied
    by the parameter vector $\theta$.
    For instance, it could be a set of **landmark constraints** that pinpoint
    the location of specific anatomical landmarks between
    the source and target shapes.


Quadratic proxies
~~~~~~~~~~~~~~~~~

Inspired by the
`XGBoost <https://xgboost.readthedocs.io/en/stable/tutorials/model.html>`_
library, we tackle the registration problem by iteratively
solving a sequence of **quadratic approximations of the objective function**.
Starting from an initial guess $\theta_0$, which usually corresponds to
the identity transformation:

.. math::

    \text{Model}(\,\theta_0 \,;\, A\,) ~=~
    A_{\theta_0} ~=~ A \qquad\text{and}\qquad
    \text{Regularization}(\theta_0) ~=~ 0~,

we **iterate until convergence** the update:

.. math::
    :label: quadratic_proxy_update
    :nowrap:

    \begin{align}
    \theta_{t+1} ~\gets~ \arg \min_{\theta} ~&
    \tfrac{1}{2}\| \theta - \target{\theta}_t\|^2_{R_t}
    ~+~
    \tfrac{1}{2}\| M_t \theta - \target{X}_t\|^2_{L_t}
    \\
    &\text{subject to}~~~ \t{C}_t \theta = \target{E}_t~. \nonumber
    \end{align}


In the equation above, $\target{\theta}_t$, $R_t$, $M_t$, $\target{X}_t$, $L_t$, $C_t$ and $\target{E}_t$
are vectors and matrices that may depend on the current estimate $\theta_t$.
The notation:

.. math::

    \| x \|^2_{A} ~:=~ \text{trace}(\t{x} A x)

refers to the squared norm of the vector $x$ with respect to the positive (semi-)definite matrix $A$.
We now introduce every term of this equation.


Regularization term - left side
---------------------------------

Given a current parameter value $\theta_t$,
we approximate the regularization term as:

.. math::

    \text{Regularization}(\theta) ~\simeq~
    \tfrac{1}{2}\| \theta - \target{\theta}_t\|^2_{R_t}

where:

  - $\target{\theta}_t$ is the target parameter estimated from our current estimate $\theta_t$,
    a vector of length P.
    It is often equal to a **vector of zeroes**, or to the projection of $\theta_t$
    on a set of "desirable" parameters, such as the group of rotation matrices.

  - $R_t$ is a P-by-P positive (semi-)definite matrix that penalizes
    the deviation between the current guess $\theta$ and the target $\target{\theta}$.
    It typically puts more emphasis on the high frequencies of the deformation
    to **penalize tears and topological changes**.



Data attachment term - right side
-------------------------------------

Inspired by the Gauss-Newton and Levenberg-Marquardt methods,
we **linearize the deformation model** around the current estimate $\theta_t$
but use a **quadratic approximation of the loss function**.
We refer to Section 17.2.2 of
`the Elements of Differentiable Programming <https://arxiv.org/abs/2403.14606>`_
for an in-depth discussion.

More precisely, given a current parameter value $\theta_t$,
we approximate the deformation model as:

.. math::

    \text{Model}(\,\theta \,;\, A\,) ~&\simeq~
    \underbrace{\text{Model}(\,\theta_t \,;\, A\,)}_{X_t}
    ~+~
    \underbrace{\text{d}_\theta\text{Model}(\,\theta_t \,;\, A\,)}_{M_t} \cdot (\theta - \theta_t) \\
    &=~
    X_t - M_t \theta_t + M_t \theta \\

Likewise, we approximate the loss function as:

.. math::

    \text{Loss}(\,X, ~B\,) ~\simeq~
    \tfrac{1}{2}\| X - \widetilde{X}_t\|^2_{L_t} .

In the equation above:

  - $X$ represents a **vector of features** of length F
    that characterizes the shape $A_{\theta} = \text{Model}(\,\theta \,;\, A\,)$.
    Usually, this is simply a **flattened vector of coordinates** $xyz$
    and F is equal to three times the number of points in the source shape A.

  - $\widetilde{X}_t$ is a vector of target shape features of length F.

  - $L_t$ is a F-by-F positive (semi-)definite matrix that penalizes
    the deviation between the current guess $X$ and the target $\widetilde{X}$.
    It is typically **(block-)diagonal** and puts more weight on points
    that are known with high confidence, such as landmarks.

Combining the two steps above, we obtain the following approximation
for the second term of the objective function:

.. math::

    \text{Loss}(\,\text{Model}(\,\theta \,;\, A\,),~ B\,) ~&\simeq~
    \tfrac{1}{2}\| X_t - M_t \theta_t + M_t \theta - \widetilde{X}_t\|^2_{L_t}
    \\
    &=~
    \tfrac{1}{2}\| M_t \theta -
    \underbrace{(\widetilde{X}_t - X_t + M_t \theta_t)}_{\target{X}_t}\|^2_{L_t}~.

In this formula, the **Loss** function is responsible for providing
the target vector of features $\widetilde{X}_t$
and the quadratic metric $L_t$
while the deformation **Model** provides
the current guess $X_t$ and the differential $M_t$.



Constraints
-------------

Finally, we linearize the constraints around the current estimate $\theta_t$:

.. math::

    \text{Constraints}(\theta) ~&\simeq~
    \text{Constraints}(\theta_t) +
    \underbrace{\text{d}_\theta\text{Constraints}(\theta_t)}_{\t{C}_t} \cdot (\theta - \theta_t) \\
    &=~
    \text{Constraints}(\theta_t) - \t{C}_t \theta_t + \t{C}_t \theta ~.

We then approximate the constraint:

.. math::

    \text{Constraints}(\theta) = E

with the linear equation:

.. math::

    E &= \text{Constraints}(\theta_t) - \t{C}_t \theta_t + \t{C}_t \theta \\
    \Longleftrightarrow~~
    \t{C}_t \theta &=
    \underbrace{E + \t{C}_t \theta_t - \text{Constraints}(\theta_t)}_{\target{E}_t}~.

The constraint matrix $C_t$ usually has **few columns**:
for instance, one for each landmark that must be preserved.



Closed-form solutions
~~~~~~~~~~~~~~~~~~~~~

In full generality, the solution of :eq:`quadratic_proxy_update`


Practical implementations
~~~~~~~~~~~~~~~~~~~~~~~~~

Deformation models
-------------------


Loss functions
---------------



Standard methods
~~~~~~~~~~~~~~~~~

Blabla
