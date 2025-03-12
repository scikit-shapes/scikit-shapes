.. _explanation_registration:

Overview of the registration pipeline
=====================================

The registration loop: iterate


.. math::

    \theta_{t+1} ~\gets~ \arg \min_{\theta} ~&
    \text{Regularization}(\theta) ~+~ \text{Loss}(\,\text{Model}(\,\theta \,;\, A\,),~ B\,)
    \\
    \text{subject to}~~~&\text{Constraints}(\theta) = E~.


Quadratic proxies
~~~~~~~~~~~~~~~~~

We approximate this as:

.. math::

    \theta_{t+1} ~\gets~ \arg \min_{\theta} ~&
    \tfrac{1}{2}\| \theta - \target{\theta}_t\|^2_{R_t}
    ~+~
    \tfrac{1}{2}\| M_t \theta - \target{X}_t\|^2_{L_t}
    \\
    \text{subject to}~~~& \t{C}_t \theta = E_t~.

In the equation above, $\target{\theta}_t$, .
We discuss .


Regularization term
----------------------

More precisely, given a current parameter value $\theta_t$,
we approximate the regularization term as:

.. math::

    \text{Regularization}(\theta) ~&\simeq~
    \tfrac{1}{2}\| \theta - \target{\theta}_t\|^2_{R_t} \\
    &=~
    \tfrac{1}{2} \text{trace}\big(
    (\theta - \target{\theta}_t)^\intercal R_t (\theta - \target{\theta}_t)
    \big)

where:

  - $\target{\theta}_t$ is the target parameter estimated from our current estimate $\theta_t$,
    a vector of length P.
    It is often equal to a **vector of zeroes**, or to the projection of $\theta_t$
    on a set of "desirable" parameters, such as the group of rotation matrices.

  - $R_t$ is a P-by-P positive (semi-)definite matrix that penalizes
    the deviation between the current guess $\theta$ and the target $\target{\theta}$.
    It typically puts more emphasis on the high frequencies of the deformation
    to **penalize tears and topological changes**.



Data attachment term
-------------------------

Inspired by the Gauss-Newton and Levenberg-Marquardt methods,
we **linearize the model** around the current estimate $\theta_t$
but use a **quadratic approximation of the loss function**.
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

  - $X$ represents a vector of features of length F
    that characterizes the shape $\text{Model}(\,\theta \,;\, A\,)$.
    Usually, this is simply a **vector of coordinates** $xyz$
    and F is equal to three times the number of points in the shape A.

  - $\widetilde{X}_t$ is a vector of target shape features of length F.

  - $L_t$ is a F-by-F positive (semi-)definite matrix that penalizes
    the deviation between the current guess $X$ and the target $\widetilde{X}$.
    It is typically **(block-)diagonal** and puts more weight on points
    that are known with high confidence, such as landmarks.

Combining the two steps above, we obtain the following approximation:

.. math::

    \text{Loss}(\,\text{Model}(\,\theta \,;\, A\,),~ B\,) ~&\simeq~
    \tfrac{1}{2}\| X_t - M_t \theta_t + M_t \theta - \widetilde{X}_t\|^2_{L_t}
    \\
    &=~
    \tfrac{1}{2}\| M_t \theta -
    \underbrace{(\widetilde{X}_t - X_t + M_t \theta_t)}_{\target{X}_t}\|^2_{L_t}~.

In this formula, the loss function is responsible for providing
the target vector of features $\widetilde{X}_t$
and the quadratic metric $L_t$
while the deformation model provides
the current guess $X_t$ and the differential $M_t$.



Closed-form solutions
~~~~~~~~~~~~~~~~~~~~~




Practical implementations
~~~~~~~~~~~~~~~~~~~~~~~~~

Deformation models
-------------------


Loss functions
---------------



Standard methods
~~~~~~~~~~~~~~~~~

Blabla
