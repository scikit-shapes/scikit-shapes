from functools import partial

from pydantic import BaseModel

from ..input_validation import typecheck
from ..linear_operators import LinearOperator
from ..types import (
    Callable,
    Function,
    Literal,
    Measure,
)
from .spectrum import Spectrum


class DiffusionParameters(BaseModel):
    method: Literal["implicit euler", "exponential", "spectral"]


class Neighborhoods:
    r"""High-level abstraction for neighborhood structures on discrete domains.

    This is the common interface for all computations that deal with **pairwise interactions**
    and distances between points: Gaussian convolutions, Laplacians, etc.

    We use the following notations:

    - $X = (x_1, \dots, x_N)$ is a **discrete domain** sampled with $N$ points.
    - $F$ is the dimension of the **feature vectors** associated to each point:
      $F=1$ for scalar functions, $F=3$ for 3D vector fields, etc.
    - $\mathcal{F} : X \rightarrow \mathbb{R}^F$ is the space of **functions** defined on
      the domain $X$
      with values of dimension $F$. We identify functions $f$ in $\mathcal{F}$
      with collections of values $(f(x_1), \dots, f(x_N))$ in $\mathbb{R}^{N\times F}$.
    - $\mathcal{F}^\star$ is the space of linear functionals on $\mathcal{F}$.
      It should be understood as a space of **measures** on $X$
      but in the discrete setting, we also identify $\mathcal{F}^\star$ with $\mathbb{R}^{N\times F}$.

    .. warning::

        Making a **clear distinction** between $\mathcal{F}$ and $\mathcal{F}^\star$
        is important to avoid confusion between functions and measures, which play
        different roles in shape analysis. However, in practice,
        both **functions** $f$ in $\mathcal{F}$ and **measures** $\alpha$ in $\mathcal{F}^\star$
        are encoded as $N$-by-$F$ arrays and the duality bracket
        is implemented as a dot product:

        .. math::

            \alpha(f)
            ~=~ \langle \alpha, f \rangle
            ~=~ \langle f, \alpha \rangle
            ~=~ \text{trace}(\alpha^\top f)
            ~=~ \sum_{i=1}^N \alpha(x_i) \cdot f(x_i)~.

    A **neighborhood structure** on the domain $X$ provides a collection of **linear operators**
    that we use to manipulate functions defined on the domain.
    The main operators are:

    - The **mass** matrix :math:`M : \mathcal{F} \rightarrow \mathcal{F}^\star`.
    - The **metric** :math:`G : \mathcal{F} \rightarrow \mathcal{F}^\star`.
    - The **cometric** :math:`K = G^{-1}: \mathcal{F}^\star \rightarrow \mathcal{F}`.

    Under the hood, these three operators are encoded as
    **symmetric** $(NF)$-by-$(NF)$ matrices
    or equivalent data structures that enable efficient matrix-vector products:

    .. math::

        M^\top = M~,~~ G^\top = G~~\text{and}~~ K^\top = K~.

    They are also **positive semi-definite**. For all function $f$ in $\mathcal{F}$
    and measure $\alpha$ in $\mathcal{F}^\star$:

    .. math::

        \langle f, Mf \rangle \geqslant 0~,~
        \langle f, Gf \rangle \geqslant 0~\text{and}~
        \langle \alpha, K\alpha \rangle \geqslant 0~.

    Intuitively, the **mass matrix** $M$ encodes the volume associated to each point of the domain
    and allows us to define the squared Euclidean norm of functions $f$ in $\mathcal{F}$ as:

    .. math::

        \|f\|^2_M ~=~ \langle f, Mf \rangle~.

    The dot product between two functions $f$ and $g$ in $\mathcal{F}$ is then defined as:

    .. math::

        \langle f, g \rangle_M ~=~ \langle f, Mg \rangle~.

    Meanwhile, the **metric** $G$ encodes the geometry of the domain via a
    norm that should penalize large gradients:

    .. math::

        \|f\|^2_G ~=~ \langle f, Gf \rangle~.

    A common choice is to use a Sobolev-type metric induced by a differential operator $\nabla$:

    .. math::

        \|f\|^2_G ~=~ \|f\|^2_M + \lambda \|\nabla f\|^2_M~,

    but other formulas are possible. Finally, the **cometric** $K$ is simply the inverse of the metric $G$.

    .. note::

        Some classical algorithms define the metric $G$ as a sparse graph Laplacian
        while others put the emphasis on the cometric $K$ encoded as a Gaussian kernel matrix.
        Instances of :class:`Neighborhoods` provide both operators
        and handle the choice of efficient linear algebra solvers internally.

    Using the mass $M$, metric $G$ and cometric $K$, we can derive
    two linear operators that act on functions $f$ in $\mathcal{F}$:

    - The **Laplace-Beltrami** operator :math:`\Delta = M^{-1}G : \mathcal{F} \rightarrow \mathcal{F}`
      is a high-pass filter.
    - The **smoothing** operator :math:`S = \Delta^{-1} = KM : \mathcal{F} \rightarrow \mathcal{F}`
      is a low-pass filter that corresponds to the resolution of
      `Poisson's equation <https://en.wikipedia.org/wiki/Poisson%27s_equation>`__.

    Finally, we also consider a **diffusion** operator :math:`Q : \mathcal{F} \rightarrow \mathcal{F}`.
    It is often defined as:

    - :math:`Q = \exp(-\Delta)`, for genuine heat diffusion.
    - :math:`Q = (I + \Delta)^{-1}` for implicit Euler integration of the heat equation,
      where $I$ is the identity operator on $\mathcal{F}$.

    The Laplacian $\Delta$, smoothing $S$ and diffusion $Q$ are all symmmetric with respect to the mass matrix $M$.
    This means that for all functions $f$ and $g$ in $\mathcal{F}$, we have:

    .. math::

        \langle \Delta f, g \rangle_M = \langle f, \Delta g \rangle_M~,~~
        \langle Sf, g \rangle_M = \langle f, Sg \rangle_M~\text{ and }~
        \langle Qf, g \rangle_M = \langle f, Qg \rangle_M~.

    In matrix form, this translates to:

    .. math::

        \Delta^\top M = M \Delta~,~~ S^\top M = MS~\text{ and }~ Q^\top M = MQ~.

    Parameters
    ----------
    mass
        The mass matrix :math:`M : \mathcal{F} \rightarrow \mathcal{F}^\star`.
        It should correspond to a symmetric matrix, i.e. $M^\top = M$.
    metric
        The metric :math:`G : \mathcal{F} \rightarrow \mathcal{F}^\star`.
        It should correspond to a symmetric matrix, i.e. $G^\top = G$.
    cometric
        The cometric :math:`K = G^{-1} : \mathcal{F}^\star \rightarrow \mathcal{F}`.
        It should correspond to a symmetric matrix, i.e. $K^\top = K$.
    diffusion
        The diffusion operator :math:`Q : \mathcal{F} \rightarrow \mathcal{F}`.
        It should be symmetric with respect to the mass matrix $M$, i.e. $Q^\top M = M Q$.
        It is often defined as:

        - :math:`Q = \exp(-\Delta)`, for genuine heat diffusion.
        - :math:`Q = (I + \Delta)^{-1}` for implicit Euler integration of the heat equation.

    spectrum
        A callable that takes an integer `n_modes` as input
        and returns a :class:`Spectrum<skshapes.Spectrum>` object
        that contains the first `n_modes` eigenvectors and eigenvalues
        of the Laplacian, smoothing and diffusion operators.

    """

    @typecheck
    def __init__(
        self,
        *,
        mass: LinearOperator[Function, Measure],
        metric: LinearOperator[Function, Measure],
        cometric: LinearOperator[Measure, Function],
        diffusion: LinearOperator[Function, Function],
        spectrum: Callable[[int], Spectrum],
    ):
        self._mass = mass
        self._metric = metric
        self._cometric = cometric
        self._diffusion = diffusion
        self._spectrum = spectrum

    @staticmethod
    def from_metric(
        *,
        mass: LinearOperator[Function, Measure],
        metric: LinearOperator[Function, Measure],
        diffusion_method: Literal["implicit euler", "exponential"],
    ):
        r"""Alternate constructor of :class:`Neighborhoods` from a mass :math:`M` and metric :math:`G`.

        Parameters
        ----------
        mass
            The mass matrix :math:`M : \mathcal{F} \rightarrow \mathcal{F}^\star`.
            It should correspond to a symmetric matrix, i.e. $M^\top = M$.
        metric
            The metric :math:`G : \mathcal{F} \rightarrow \mathcal{F}^\star`.
            It should correspond to a symmetric matrix, i.e. $G^\top = G$.
        diffusion_method
            The method used to define the diffusion operator :math:`Q`:

            - ``"implicit euler"`` to define :math:`Q = (I + \Delta)^{-1} = (M + G)^{-1} M`.
            - ``"exponential"`` to define :math:`Q = \exp(-\Delta) = \exp(-M^{-1}G)`.

        """
        M = mass
        G = metric
        K = G.inverse()

        if diffusion_method == "implicit euler":
            # Use the identity
            # Q = (I + Delta)^(-1)
            #   = (I + M^(-1) G)^(-1)
            #   = (M + G)^(-1) M
            # M + G is symmetric positive semi-definite, so this is easy to solve.
            Q = (M + G).inverse() @ M

        elif diffusion_method == "exponential":
            # Use a routine such as scipy.sparse.linalg.expm_multiply,
            # depending on the sparsity of M and G.
            Q = (-M.inverse() @ G).matrix_exp()

        return Neighborhoods(
            mass=M,
            metric=G,
            cometric=K,
            diffusion=Q,
            spectrum=partial(
                Spectrum.from_metric,
                mass=M,
                metric=G,
                diffusion_method=diffusion_method,
            ),
        )

    @staticmethod
    def from_cometric(
        *,
        mass: LinearOperator[Function, Measure],
        cometric: LinearOperator[Function, Measure],
        diffusion_method: Literal["implicit euler", "exponential"],
    ):
        r"""Alternate constructor of :class:`Neighborhoods` from a mass :math:`M` and cometric :math:`K`.

        Parameters
        ----------
        mass
            The mass matrix :math:`M : \mathcal{F} \rightarrow \mathcal{F}^\star`.
            It should correspond to a symmetric matrix, i.e. $M^\top = M$.
        cometric
            The cometric :math:`K : \mathcal{F}^\star \rightarrow \mathcal{F}`.
            It should correspond to a symmetric matrix, i.e. $K^\top = K$.
        diffusion_method
            The method used to define the diffusion operator :math:`Q`:

            - ``"implicit euler"`` to define :math:`Q = (I + \Delta)^{-1} = K (M^{-1} + K)^{-1}`.
            - ``"exponential"`` to define :math:`Q = \exp(-\Delta) = \exp(-M^{-1}K^{-1})`.

        """

        M = mass
        K = cometric
        G = K.inverse()

        if diffusion_method == "implicit euler":
            # Use the identity
            # Q = (I + Delta)^(-1)
            #   = (I + M^(-1) K^(-1))^(-1)
            #   = K (M^(-1) + K)^(-1)
            # M^(-1) + K is symmetric positive semi-definite, so this is easy to solve.
            Q = K @ (M.inverse() + K).inverse()

        elif diffusion_method == "exponential":
            # Use a routine such as scipy.sparse.linalg.expm_multiply,
            # depending on the sparsity of M and G.
            # Unlike what we have for implicit euler, there is no simple
            # expression of Q in terms of K and M.
            Q = (-M.inverse() @ G).matrix_exp()

        return Neighborhoods(
            mass=M,
            metric=G,
            cometric=K,
            diffusion=Q,
            spectrum=partial(
                Spectrum.from_cometric,
                mass=M,
                cometric=K,
                diffusion_method=diffusion_method,
            ),
        )

    @property
    @typecheck
    def mass(self) -> LinearOperator[Function, Measure]:
        r"""mass(self)
        The mass matrix :math:`M : \mathcal{F} \rightarrow \mathcal{F}^\star`.

        It corresponds to a symmetric matrix, i.e. $M^\top = M$.
        """
        return self._mass

    @property
    @typecheck
    def metric(self) -> LinearOperator[Function, Measure]:
        r"""The metric :math:`G : \mathcal{F} \rightarrow \mathcal{F}^\star`.

        It corresponds to a symmetric matrix, i.e. $G^\top = G$.
        """
        return self._metric

    @property
    @typecheck
    def cometric(self) -> LinearOperator[Measure, Function]:
        r"""The cometric :math:`K = G^{-1} : \mathcal{F}^\star \rightarrow \mathcal{F}`.

        It corresponds to a symmetric matrix, i.e. $K^\top = K$.
        """
        return self._cometric

    @property
    @typecheck
    def diffusion(self) -> LinearOperator[Function, Function]:
        r"""The diffusion operator :math:`Q : \mathcal{F} \rightarrow \mathcal{F}`.

        It is symmetric with respect to the mass matrix $M$, i.e. $Q^\top M = M Q$.
        It is often defined as:

        - :math:`Q = \exp(-\Delta)`, for genuine heat diffusion.
        - :math:`Q = (I + \Delta)^{-1}` for implicit Euler integration of the heat equation.

        """
        return self._diffusion

    @property
    @typecheck
    def smoothing(self) -> LinearOperator[Function, Function]:
        r"""The smoothing operator :math:`S = KM : \mathcal{F} \rightarrow \mathcal{F}`.

        It is symmetric with respect to the mass matrix $M$, i.e. $S^\top M = M S$.
        """
        return self.cometric @ self.mass

    @property
    @typecheck
    def laplacian(self) -> LinearOperator[Function, Function]:
        r"""The Laplace-Beltrami operator :math:`\Delta = M^{-1}G : \mathcal{F} \rightarrow \mathcal{F}`.

        It is symmetric with respect to the mass matrix $M$, i.e. $\Delta^\top M = M \Delta$.
        """
        # Currently, we only support (block-)diagonal mass matrices,
        # whose inverses are easy to compute.
        return self.mass.inverse() @ self.metric

    @typecheck
    def spectrum(self, n_modes: int) -> Spectrum:
        r"""The first :math:`R` eigenvectors and eigenvalues of the underlying operators.

        We return a :class:`Spectrum<skshapes.Spectrum>` object that contains a
        collection of $R$ functions $\phi_1, \dots, \phi_R$
        encoded as a $(R, N, F)$ Tensor that we understand below
        as a $R$-by-$NF$ matrix $\Phi = [\phi_1| \cdots | \phi_R]$,
        where $N$ is the number of sampling points on the domain $X$
        and $F$ is the dimension of the feature vectors handled by this
        :class:`Neighborhoods` structure.
        This family is orthonormal for the dot product induced by the symmetric mass matrix $M$, so that:

        .. math::

            \Phi^\top M \Phi = I_R~.

        The output also contains vectors of eigenvalues that correspond to
        the functional operators $\Delta, S, Q : \mathcal{F} \rightarrow \mathcal{F}$.

        The vector of :attr:`laplacian_eigenvalues` $\Lambda^\Delta = (\lambda^\Delta_1, \dots, \lambda^\Delta_R)$
        corresponds to the Laplacian $\Delta = M^{-1}G$
        implemented by :attr:`laplacian`:

        .. math::

            \Delta \phi_i = \lambda^\Delta_i \phi_i

        so that:

        .. math::

            \Delta \simeq  \Phi \, \text{diag}(\Lambda^\Delta) \,\Phi^\top M~.

        Likewise, the vector of :attr:`smoothing_eigenvalues`
        $\Lambda^S = (\lambda^S_1, \dots, \lambda^S_R)$
        corresponds to the smoothing operator $S = K M$
        implemented by :attr:`smoothing`:

        .. math::

            S \phi_i = \lambda^S_i \phi_i~\text{ so that }~
            S \simeq \Phi \,\text{diag}(\Lambda^S) \,\Phi^\top M~.


        The vector of :attr:`diffusion_eigenvalues`
        $\Lambda^Q = (\lambda^Q_1, \dots, \lambda^Q_R)$
        corresponds to the diffusion operator $Q$
        implemented by :attr:`diffusion`:

        .. math::

            Q \phi_i = \lambda^Q_i \phi_i

        so that:

        .. math::

            Q \simeq \Phi \,\text{diag}(\Lambda^Q) \,\Phi^\top M~.

        We typically have:

        .. math::

            \lambda^S_i = 1 / \lambda^\Delta_i~\text{ and }~ \lambda^Q_i = \exp(-\lambda^\Delta_i)~,
            ~\text{ or }~ \lambda^Q_i = 1 / (1 + \lambda^\Delta_i)~.


        Parameters
        ----------
        n_modes
            The number of eigenmodes $R$ to compute.

        Returns
        -------
        Spectrum
            A :class:`Spectrum<skshapes.Spectrum>` object, as described above.


        """
        return self._spectrum(n_modes=n_modes)
