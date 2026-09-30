.. _explanation_moments:

Computing local moments
========================

As a pre-processing for shape analysis and registration,
one often needs to compute local features such as point normals
and curvatures.
In this context, local point moments are a useful building block.
We now explain in depth the implementation of our
:meth:`~skshapes.PolyData.point_moments` method.

Definition
----------

Let us consider a point cloud :math:`x_1, \dots, x_N` in dimension :math:`D = 2` or :math:`3`.
We associate to every point :math:`x_i` a local neighborhood :math:`\nu_i(x)`,
understood as a distribution of mass on the :math:`N` points.
The full :meth:`~skshapes.PolyData.point_neighborhoods` structure is encoded
in the :math:`N \times N` matrix:

.. math::

    \mathcal{V} ~=~
    \left(
    \begin{array}{ccc}
    ~ & \nu_1 & ~ \\ \hline
    ~ & \vdots & \\ \hline
    ~ & \nu_N & ~
    \end{array}
    \right)
    ~=~
    \begin{pmatrix}
    \nu_1(x_1) & \dots & \nu_1(x_N) \\
    \vdots & \ddots & \vdots \\
    \nu_N(x_1) & \dots & \nu_N(x_N)
    \end{pmatrix}~.

**Order 0.** The local mass :math:`m_i` around point :math:`x_i` is defined by:

.. math::

    m_i ~=~ \sum_{j=1}^N \nu_i(x_j)~.

**Order 1.** The local mean :math:`\overline{x}_i` around point :math:`x_i` is defined by:

.. math::

    \overline{x}_i ~=~ \frac{1}{m_i} \sum_{j=1}^N \nu_i(x_j) x_j~.

**Order 2.** The local covariance matrix :math:`\Sigma_i` around point :math:`x_i` is defined by:

.. math::

    \Sigma_i ~=~ \frac{1}{m_i} \sum_{j=1}^N \nu_i(x_j) \, (x_j - \overline{x}_i) (x_j - \overline{x}_i)^\intercal~.

Just like other
`feature transforms <https://en.wikipedia.org/wiki/Scale-invariant_feature_transform>`_
or
`shape contexts <https://en.wikipedia.org/wiki/Shape_context>`_,
point moments encode some information
about the local geometry of the shape. They are popular in shape analysis as precursors
to rotation-invariant features:
see the vast literature on the
`Riesz transform <https://en.wikipedia.org/wiki/Riesz_transform>`_,
`steerable filters <https://en.wikipedia.org/wiki/Steerable_filter>`_
and
`Hu's geometric moment invariants <https://en.wikipedia.org/wiki/Image_moment>`_.

Implementation
--------------

**Local masses and means.**
Moments of order 0 and 1 are easy to compute using matrix-matrix products with the
neighborhood operator :math:`\mathcal{V}`. The vector of local masses :math:`m = (m_1, \dots, m_N)` is given by:

.. math::

    m
    ~=~ \begin{pmatrix} m_1 \\ \vdots \\ m_N \end{pmatrix}
    ~=~ \mathcal{V} \begin{pmatrix} 1 \\ \vdots \\ 1 \end{pmatrix}~,

and the :math:`N\times D` matrix of local means :math:`(\overline{x}_1, \dots, \overline{x}_N)` is given by:

.. math::


    \left(
    \begin{array}{ccc}
    ~ & \overline{x}_1 & ~ \\ \hline
    ~ & \vdots & \\ \hline
    ~ & \overline{x}_N & ~
    \end{array}
    \right)
    ~=~ \frac{1}{m} \mathcal{V}
    \left(
    \begin{array}{ccc}
    ~ & x_1 & ~ \\ \hline
    ~ & \vdots & \\ \hline
    ~ & x_N & ~
    \end{array}
    \right)~.

On the other hand, computing local covariances at scale is surprisingly tricky.


**Local covariances.**  To compute the local covariance matrix :math:`\Sigma_i` around point :math:`x_i`,
performing the summation:

.. math::

    \Sigma_i ~=~ \frac{1}{m_i} \sum_{j=1}^N \nu_i(x_j) \,(x_j - \overline{x}_i) (x_j - \overline{x}_i)^\intercal

in parallel over the indices :math:`i` is do-able if the neighborhood weight
:math:`\nu_i(x_j)` is known in closed form.
For instance, if we use a Gaussian window of radius :math:`\sigma`:

.. math::

    \nu_i(x_j) ~=~ \exp\left(-\tfrac{1}{2\sigma^2} \|x_i - x_j\|^2\right)~,

the
`KeOps <https://www.kernel-operations.io/>`_
library performs the summation in fractions of a second on a GPU,
even for large clouds of :math:`N > 100k` points.

**Limitations of the brute force approach.**
Unfortunately, this approach is not tractable in all settings:

- The
  `Nystroem approximation <https://en.wikipedia.org/wiki/Low-rank_matrix_approximations#Nystr%C3%B6m_approximation>`_,
  the
  `Fast Multipole Method <https://en.wikipedia.org/wiki/Fast_multipole_method>`_
  or
  `DiffusionNet <https://arxiv.org/abs/2012.00888>`_
  layers
  rely on **low-rank approximations** of the neighborhood matrix :math:`\mathcal{V}`.

- The
  `heat method <https://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/>`_
  for geodesic distance computations relies on an **implicit** definition.
  The neighborhood operator :math:`\mathcal{V}` is defined as:

    .. math::

        \mathcal{V} ~=~ (\text{Id}_{N} - \tfrac{\sigma^2}{2}\Delta)^{-1}~
        \simeq~ \exp(\tfrac{\sigma^2}{2}\Delta)~,

  where :math:`\Delta` is a discrete
  :math:`N\times N` Laplacian matrix
  and matrix-matrix products with :math:`\mathcal{V}` are implemented via
  a sparse Cholesky factorization.

In both settings, while matrix multiplication with :math:`\mathcal{V}` is cheap,
accessing all of the :math:`N^2` entries
:math:`\nu_i(x_j)` of :math:`\mathcal{V}` is not
tractable when :math:`N > 10k`.

Naive quadratic expansion
-------------------------

**Expanding the squared difference.**
A simple work-around would be to remark that:


.. math::

    \Sigma_i ~
    &=~ \frac{1}{m_i} \sum_{j=1}^N \nu_i(x_j) \,(x_j - \overline{x}_i) (x_j - \overline{x}_i)^\intercal \\
    &=~ \frac{1}{m_i} \sum_{j=1}^N \nu_i(x_j) \, x_j x_j^\intercal - \overline{x}_i \overline{x}_i^\intercal~.

This is the `standard statistical formula <https://en.wikipedia.org/wiki/Covariance#Auto-covariance_matrix_of_real_random_vectors>`_:

.. math::

    \text{Var}(X)~=~ \mathbb{E}(X^2) - \mathbb{E}(X)^2~\geqslant ~0~,

which quantifies the spread of a random variable :math:`X \sim \nu_i`
by computing the gap in
`Jensen's inequality <https://en.wikipedia.org/wiki/Jensen%27s_inequality>`_
when applied to the convex function :math:`x \mapsto x^2`.

**Numerical accuracy.**
Theoretically, we could use this identity to compute the
:math:`D\times D` covariance
matrices :math:`\Sigma_i` efficiently with:

.. math::
    \left(
    \begin{array}{ccc}
    ~ & \Sigma_1 & ~ \\ \hline
    ~ & \vdots & \\ \hline
    ~ & \Sigma_N & ~
    \end{array}
    \right)
    ~=~
    \frac{1}{m} \mathcal{V}
    \left(
    \begin{array}{ccc}
    ~ & x_1 x_1^\intercal & ~ \\ \hline
    ~ & \vdots & \\ \hline
    ~ & x_N x_N^\intercal & ~
    \end{array}
    \right)
    ~-~
    \left(
    \begin{array}{ccc}
    ~ & \overline{x}_1 \overline{x}_1^\intercal & ~ \\ \hline
    ~ & \vdots & \\ \hline
    ~ & \overline{x}_N \overline{x}_N^\intercal & ~
    \end{array}
    \right)~.


Unfortunately, this formula runs into
`catastrophic cancellation <https://en.wikipedia.org/wiki/Catastrophic_cancellation>`_
when it is implemented using
`floating-point arithmetic <https://en.wikipedia.org/wiki/Floating-point_arithmetic>`_.
With neighborhoods :math:`\nu_i` defined at scale :math:`\sigma > 0`,
the coefficients of the covariance matrix :math:`\Sigma_i` are typically of order :math:`\sigma^2`,
while :math:`x_jx_j^\intercal`
and :math:`\overline{x}_i\overline{x}_i^\intercal`
are of order :math:`\max(\|x_i\|)^2`.

In common shape processing scenarios, even if the point cloud
:math:`(x_1, \dots, x_N)` is centered and normalized,
the neighborhood scale :math:`\sigma` is 100 to 1,000 times
smaller than the diameter of the shape.
We are interested in fine details and don't want to over-smooth our shapes!

As a consequence, the coefficients of the covariance matrix :math:`\Sigma_i` are usually
**4 to 6 orders of magnitude smaller** than both terms in the difference,
which scale like :math:`\|x_i\|^2`.
They cannot be computed reliably from the identity above
if the :math:`x_jx_j^\intercal`
and :math:`\overline{x}_i\overline{x}_i^\intercal` are
stored in single-precision
`float32 <https://en.wikipedia.org/wiki/Single-precision_floating-point_format>`_ format
with 7 significant digits in base 10.

**Portability.**
Performing the computation in double-precision
`float64 <https://en.wikipedia.org/wiki/Double-precision_floating-point_format>`_,
including the matrix multiply with the neighborhood operator :math:`\mathcal{V}`,
could provide a sufficient accuracy.
However, while float64 arithmetic is supported by most CPUs and high-end GPUs,
it is **not implemented** at the hardware level by consumer-grade GPUs.
A typical "gaming" GPU such as a
`Nvidia RTX4090 <https://www.techpowerup.com/gpu-specs/geforce-rtx-4090.c3889>`_
experiences a **x64 slow-down** when switching from float32 to float64 arithmetic.


Stable trigonometric expansion
------------------------------

**Approximating x with sin(x).**
To provide a fast, general and portable implementation of point moments,
we rely on the fact that if :math:`x_j - \overline{x}_i` is of order
:math:`\sigma` and :math:`\ell = 10\, \sigma`, then:

.. math::
    x_j - \overline{x}_i~
    &\simeq~ \ell \, \sin \left( \frac{x_j - \overline{x}_i}{\ell} \right) \\
    &=~\ell\, \sin(X_j - \overline{X}_i)~,

where
:math:`X_j = x_j / \ell` and :math:`\overline{X}_i = \overline{x}_i / \ell`.

This implies that if the support of the neighborhood
:math:`\nu_i:x_j\mapsto \nu_i(x_j)` of point :math:`x_i`
is concentrated on points :math:`x_j` which are at distance
at most :math:`5\,\sigma` of :math:`x_i`,
then:

.. math::
    \Sigma_i~
    &=~\frac{1}{m_i} \sum_{j=1}^N \nu_i(x_j) \,(x_j - \overline{x}_i) (x_j - \overline{x}_i)^\intercal \\
    &\simeq~ \frac{\ell^2}{m_i}
    \sum_{j=1}^N \nu_i(x_j) \,\sin(X_j - \overline{X}_i) \sin(X_j - \overline{X}_i)^\intercal~.

**Trigonometric expansion.**
Denoting by :math:`\imath` the imaginary unit
and :math:`\overline{z}` the complex conjugate of :math:`z`,
standard trigonometric identities let us rewrite this equation as:

.. math::
    \Sigma_i~
    \simeq~ \frac{\ell^2}{m_i}
    \frac{1}{(2\imath)^2}
    \sum_{j=1}^N \nu_i(x_j) \,(z_{ij} - \overline{z}_{ij}) (z_{ij}^\intercal - \overline{z}_{ij}^\intercal)~,

where :math:`z_{ij} = \exp[\imath (X_j - \overline{X}_i)]` is a
:math:`D`-dimensional vector of unitary complex numbers.

We know that for any complex-valued vector $z$:

.. math::
    (z - \overline{z}) (z^\intercal - \overline{z}^\intercal)
    ~=~ 2 \,[
        \text{Re}(zz^\intercal)
        -
        \text{Re}(z\overline{z}^\intercal)
    ]

We can thus write the relevant terms as products of $i$ and $j$ factors:

.. math::

    z_{ij} z^\intercal_{ij}
    ~&=~
    \exp[\imath (X_j - \overline{X}_i)]\,\cdot\, \exp[\imath (X_j^\intercal - \overline{X}_i^\intercal)]\\
    ~&=~
    \exp[-\imath (\overline{X}_i + \overline{X}_i^\intercal)]\,\cdot\, \exp[\imath (X_j + X_j^\intercal)] \\
    z_{ij} \overline{z}^\intercal_{ij}
    ~&=~
    \exp[-\imath (X_j - \overline{X}_i)]\,\cdot\, \exp[\imath (X_j^\intercal - \overline{X}_i^\intercal)]\\
    ~&=~
    \exp[-\imath (\overline{X}_i - \overline{X}_i^\intercal)]\,\cdot\, \exp[\imath (X_j - X_j^\intercal)]~.

If we denote by :math:`\langle w, z^\intercal \rangle = \text{Re}(\overline{w}z^\intercal)`
the dot product applied coordinate-wise on our $2\times 2$ or $3 \times 3$ matrices,
we can now rewrite the covariance matrix $\Sigma_i$ as:

.. math::
    \Sigma_i~
    \simeq~ \frac{\ell^2}{2}
    \bigg[\,
    &\Big\langle
    \exp\big[ \imath (\overline{X}_i - \overline{X}_i^\intercal) \big],
    \frac{1}{m_i} \sum_{j=1}^N \nu_i(x_j)\,
    \exp\big[ \imath (X_j - X_j^\intercal) \big]
    \Big\rangle \\
    ~-~
    &\Big\langle
    \exp \big[\imath (\overline{X}_i + \overline{X}_i^\intercal)\big],
    \frac{1}{m_i} \sum_{j=1}^N \nu_i(x_j)\,
    \exp \big[\imath (X_j + X_j^\intercal) \big]
    \Big\rangle
    \,\bigg]


**Intuition.**
Looking at the diagonal coefficients of the covariance matrix :math:`\Sigma_i`,
we know that $\overline{X}_i = \overline{X}_i^\intercal$
and $X_j = X_j^\intercal$ with a slight abuse of notation
so the formula simply reads:

.. math::
    \text{diag}(\Sigma_i)
    ~\simeq~ \frac{\ell^2}{2}
    \bigg[\,
    1
    ~-~\Big\langle
    \exp \big[2 \imath \overline{X}_i\big],
    \frac{1}{m_i} \sum_{j=1}^N \nu_i(x_j)\,
    \exp \big[2\imath X_j \big]
    \Big\rangle
    \,\bigg]~.

This formula estimates the spread of $X_j = x_j / \ell$ in the neighborhood
of $\overline{X}_i = \overline{x}_i / \ell$ using a **generalized Jensen's inequality**
on the boundary of the (convex) unit disk of the complex plane.


**Numerical analysis.**
To perform this computation, one simply needs to use a matrix-matrix product
with the neighborhood operator :math:`\mathcal{V}` to smooth the coefficients of
the $D\times D$ tensors:

.. math::
    \cos(X_j + X_j^\intercal),
    ~~\sin(X_j + X_j^\intercal),
    ~~\cos(X_j - X_j^\intercal)~~\text{and}
    ~~\sin(X_j - X_j^\intercal)~.

The first three tensors are symmetric, while the latter is skew-symmetric.
This leaves us with $3 \cdot 3 + 1 = 10$ channels in dimension $D=2$,
and $3\cdot 6 + 3 = 21$ channels in dimension $D=3$.

Crucially, **the complex exponentials are all of order 1**.
Using this trigonometric expansion, we are estimating a quantity of order
$\sigma^2$ as the difference of two terms of order
$\ell^2 = 100\, \sigma^2$.
Even if the computation is performed in float32 arithmetic,
this still leaves approximately $7-2 = 5$ digits of decimal precision for $\Sigma_i$,
which is suitable for our purposes.
We thus favor this method by default in
:meth:`~skshapes.PolyData.point_moments` and all downstream methods.
