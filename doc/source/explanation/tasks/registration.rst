.. _explanation_registration:

Overview of the registration algorithm
======================================


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

- $\text{Constraints}(\theta) = E$ is a set of C equality constraints that
  the parameter vector $\theta$ must satisfy.
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

    \text{Loss}(\,A_\theta, ~B\,) ~\simeq~
    \tfrac{1}{2}\| X_\theta - \widetilde{X}_t\|^2_{L_t} .

In the equation above:

- $X_\theta$ represents a **vector of features** of length F
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

.. note::

    For the sake of simplicity, we now drop the $t$ indices
    from the matrices and vectors $\target{\theta}_t$, $M_t$, etc.


Closed-form solutions
~~~~~~~~~~~~~~~~~~~~~

As discussed in standard references on
`quadratic programming <https://en.wikipedia.org/wiki/Quadratic_programming#Equality_constraints>`_,
the solution $\theta_{t+1}$ of the optimization problem :eq:`quadratic_proxy_update`
is such that:

.. math::
    :label: quadratic_solution_general_case
    :nowrap:

    \begin{align}
    \begin{bmatrix}
      O & C \\
      \t{C} & 0
    \end{bmatrix}
    \begin{bmatrix}
      \theta_{t+1} \\
      \lambda
      \end{bmatrix}
      ~=~
    \begin{bmatrix}
      T \\
      \target{E}
    \end{bmatrix}
    \end{align}

where:

- $O := R + \t{M} L M$ is a P-by-P positive definite matrix: the **objective** operator.
- $T := R \target{\theta} + \t{M} L \target{X}$ is a vector of length P: the **target** values.
- $C = C_t$ is the P-by-C matrix of constraints.
- $\target{E} = \target{E}_t$ is the vector of C constraint values.
- $\lambda$ is the vector of Lagrange multipliers, of length C.

When the P-by-P objective operator $O$ and the C-by-C restriction
$(\t{C}O^{-1}C)$ are invertible, the solution simplifies to:

.. math::
    :label: quadratic_solution_invertible
    :nowrap:

    \begin{align}
    \theta_{t+1} ~\gets~
    O^{-1}~ \big(~
      T
      -
      C (\t{C}O^{-1}C)^{-1}
      (
        \t{C}O^{-1}T
        - \target{E}
      )~
    \big)
    \end{align}

When there is no constraint, we can simply compute:


.. math::
    :label: quadratic_solution_no_constraint
    :nowrap:

    \begin{align}
    \theta_{t+1} ~\gets~
    O^{-1}T~
    ~=~(R + \t{M} L M)^{-1} (R \target{\theta} + \t{M} L \target{X})~.
    \end{align}


If the target parameter $\target{\theta}$ is equal to a vector of zeroes,
this simplifies further to:

.. math::
    :label: quadratic_solution_no_constraint_zero_target
    :nowrap:

    \begin{align}
    \theta_{t+1} ~\gets~
    (R + \t{M} L M)^{-1} \t{M} L \target{X}~.
    \end{align}

Assuming that the model differential $M$ and the
loss metric $L$ are invertible, we write:

.. math::
    :label: quadratic_solution_no_constraint_zero_target_invertible
    :nowrap:

    \begin{align}
    \theta_{t+1} ~\gets~
    (L^{-1} M^{-\intercal} R + M)^{-1} \target{X}~,
    \end{align}

where $\target{X} = \widetilde{X} - X$ is the difference between the target
$\widetilde{X}_t$ and the current guess $X_t$ for the vector of features
that characterizes the deformed shape $A_{\theta_t}$.



Efficient implementations
~~~~~~~~~~~~~~~~~~~~~~~~~

Efficient shape registration is about **implementing these equations
with optimum speed and memory footprint** for different choices
of the operators $R$ (regularization), $M$ (model), $L$ (loss) and $C$ (constraints).

We split this complexity in two complementary parts:

- The **"loss"** Python module interacts with the target shape $B$.
  It is responsible for the operator $L$ and may create some constraints.
- The **deformation "model"** Python module handles the parameter vector $\theta$,
  the operators $R$ and $M$ and may create some constraints $C$.
  It is ultimately responsible for implementing the step
  from $\theta_t$ to $\theta_{t+1}$, following the most relevant
  equation from :eq:`quadratic_solution_general_case` to :eq:`quadratic_solution_no_constraint_zero_target_invertible`.


Our algorithm to register the source $A$ onto the target $B$ can then be summarized as:

- **Initialization**:

  - Let the **model** set $\theta_0$ to the identity transformation,
    i.e. start with an initial guess $A_{\theta_0}$ that is equal to the source shape $A$.

- **Iterative optimization loop**, updating $\theta_t$ to $\theta_{t+1}$ until convergence:

  - Let the **model** compute the current guess $A_{\theta_t} = \text{Model}(\theta_t\,;\,A)$.
  - Let the **loss** take as input the current guess $A_{\theta_t}$ and the target shape $B$
    to output the target vector of features $\widetilde{X}_t$ and the
    loss metric $L_t$.
  - Let the **model** compute the differential $M_t$ and the full target vector $\target{X}_t$.
  - Let the **model** compute the update $\theta_{t+1}$ from $\theta_t$.
    The choice of the most relevant equation from :eq:`quadratic_solution_general_case` to :eq:`quadratic_solution_no_constraint_zero_target_invertible`
    is dependent on the structure of the model operator $M_t$
    and the regularization metric $R_t$.

- **Finalization**:

  - Let the **model** compute the final shape $A^\star = \text{Model}(\theta^\star\,;\,A)$
    and the regularization penalty $\text{Regularization}(\theta^\star)$.
  - Let the **loss** compute the discrepancy
    $\text{Loss}(A^\star, B)$ between the final shape $A^\star$ and the target shape $B$.
  - Return the **optimal parameter** $\theta^\star$, the **final shape** $A^\star$,
    the final loss and regularization penalty to the user.


Loss functions
---------------

In this context, a loss function is a Python class that implements:

- A method :class:`loss.value(A, B)` that takes as input two shapes
  (instances of :class:`~skshapes.PolyData`)
  and returns a scalar that measures the discrepancy between the two shapes.

- A method :class:`loss.quadratic_proxy(A, B)` that takes as input two shapes
  and returns the fundamental information that is required by the optimization loop:
  a positive (semi-)definite loss metric $L$
  and a scaled vector of F target features $L\widetilde{X}$.
  A default option is to rely on the first or second derivatives of :class:`loss.value(A, B)`,
  which corresponds to implementing a semi-implicit Euler or Newton scheme
  on the full objective function.
  We typically override this with custom proxies for efficiency.


We provide a set of standard loss functions in the :mod:`skshapes.loss` module:

- The :class:`~skshapes.loss.LandmarkLoss` implements simple
  :ref:`point-wise penalties and constraints <explanation_loss_landmark>`.
  They are used to pin points that are clearly identified in both shapes.

- The :class:`~skshapes.loss.ClosestPointLoss` implements
  penalties based on
  :ref:`nearest neighbor projections <explanation_loss_closest_point>`.
  These are also known as (soft-)Hausdorff distances and correspond
  to the log-likelihoods of Gaussian Mixture models.

- The :class:`~skshapes.loss.KernelLoss` implements penalties
  based on
  :ref:`kernel convolutions <explanation_loss_kernel>`.
  These are also known as Maximum Mean Discrepancies (MMD) or
  electrostatic Coulomb penalties.

- The :class:`~skshapes.loss.OptimalTransportLoss` implements penalties
  based on
  :ref:`optimal transport correspondences <explanation_loss_optimal_transport>`.
  These are also known as Wasserstein or
  Earth Mover's Distances (EMD).


Crucially, we support **positive linear combinations** of existing
loss functions. Mathematically, this corresponds to concatenating
the constraints that come from each loss and adjusting the
formulas for the objective operator $O$ and the target vector $T$
in :eq:`quadratic_solution_general_case`.
If we consider a loss function:

.. math::

    \text{Loss}(\,A_\theta, ~B\,) ~=~ \sum_i w_i\, \text{Loss}_i(\,A_\theta, ~B\,)
    ~\simeq~ \sum_i  \tfrac{w_i}{2}\| X_\theta - \widetilde{X}_i\|^2_{L_i}~,

then the target vector $T$ and the objective operator $O$ become:

.. math::

    O ~&:=~ R +  \t{M}
    \underbrace{\big(\sum_i w_i L_i\big)}_{L_\text{sum}}
    M~,\\
    T ~&:=~ R \target{\theta} + \t{M} \big(\sum_i w_i  L_i \target{X}_i\big) \\
    &=~
    R \target{\theta} + \t{M} \big(\sum_i w_i  L_i (\widetilde{X}_i + M \theta - X)\big) \\
    &=~
    R \target{\theta} + \t{M} \big[
      \underbrace{\big(\sum_i w_i  L_i \widetilde{X}_i\big)}_{(L\widetilde{X})_\text{sum}}
      +
      L_\text{sum} (M \theta - X)
      \big]~.


Going further, you may implement a **custom loss function** by following
the :ref:`user guide <explanation_loss_custom>`.


Deformation models
-------------------

Likewise,
a deformation model is a Python class that provides:

- Access to the parameters $\theta$ as Python attributes.
  We follow the
  `scikit-learn convention <https://scikit-learn.org/stable/developers/develop.html#estimated-attributes>`_
  and name these tensors with trailing underscores:
  :attr:`model.affine_transformation_matrix_`,
  :attr:`model.control_points_` , etc.

- A method :class:`model.fit(shape)` that takes as input a source shape $A$
  (an instance of :class:`~skshapes.PolyData`) and allocates
  the relevant amount of memory for the parameters $\theta$.
  The initial value $\theta_0$ corresponds to the identity transformation,
  and the model may perform some useful pre-computations at this stage.

- A method :class:`model.transform()` that returns
  the deformed shape $A_\theta$.

- A method :class:`model.regularization_penalty()` that returns
  the current value for the regularization penalty $\text{Regularization}(\theta)$.

- A method :class:`model.optimization_step(scaled_target, weights)`
  that takes as input a vector of scaled target features $L\widetilde{X}$,
  a metric $L$ and performs the update from $\theta_t$ to $\theta_{t+1}$.
  In practice, $L\widetilde{X}$ is encoded as a N-by-F tensor
  where N denotes the number of points in the source shape $A$.
  The metric $L$ is encoded as a non-negative vector of N weights
  if it is a diagonal operator, or
  as a N-by-F-by-F collection of positive (semi-)definite matrices
  if it is block-diagonal.


We provide a set of standard deformation models in the :mod:`~skshapes.model` module:

- The :class:`~skshapes.model.AffineDeformation` model provides access to
  all methods that are parameterized by an affine transformation matrix.
  This includes:

  - **Pure translations**,
  - **Rigid transformations** (rotation and translation),
  - **Similarity transformations** (rotation, translation and isotropic scaling),
  - **Affine transformations** (rotation, translation, scaling and shearing).

- The :class:`~skshapes.model.FreeFormDeformation` model provides access to
  all methods that rely on control points.
  Mathematically speaking,  :ref:`landmarks loss <explanation_loss_landmark>` (),
  which is used to pin points that are clearly identified in both shapes.


Standard methods
~~~~~~~~~~~~~~~~~

Scikit-shapes allows you to use any deformation model with any loss function.
This allows you to reproduce reference methods such as:

  - **Thin-plate splines** with a **landmarks constraints**, as in
    `Principal warps: thin-plate splines and the decomposition of deformations <https://cseweb.ucsd.edu/classes/sp03/cse252/bookstein.pdf>`_
    by Fred L. Bookstein (1989).
