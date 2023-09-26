import torch
from ..types import (
    typecheck,
    FloatTensor,
    DoubleTensor,
    Float2dTensor,
    Double2dTensor,
    Number,
    Literal,
    Union,
    Optional,
)


def symmetric_sum(a, b):
    """Implements the symmetric terms that appear in the tensor expansion of (a+b)^n."""
    N = a.shape[0]
    D = a.shape[1]

    if a.shape == (N, D) and b.shape == (N, D, D):
        # Term 1-2 that appears in the tensor expansion of (a+b)^3
        term_1 = a.view(N, D, 1, 1) * b.view(N, 1, D, D)
        term_2 = a.view(N, 1, D, 1) * b.view(N, D, 1, D)
        term_3 = a.view(N, 1, 1, D) * b.view(N, D, D, 1)
        res = term_1 + term_2 + term_3
        assert res.shape == (N, D, D, D)
        return res

    elif a.shape == (N, D) and b.shape == (N, D, D, D):
        # Term 1-4 that appears in the tensor expansion of (a+b)^4
        term_1 = a.view(N, D, 1, 1, 1) * b.view(N, 1, D, D, D)
        term_2 = a.view(N, 1, D, 1, 1) * b.view(N, D, 1, D, D)
        term_3 = a.view(N, 1, 1, D, 1) * b.view(N, D, D, 1, D)
        term_4 = a.view(N, 1, 1, 1, D) * b.view(N, D, D, D, 1)
        res = term_1 + term_2 + term_3 + term_4
        assert res.shape == (N, D, D, D, D)
        return res

    elif a.shape == (N, D, D) and b.shape == (N, D, D):
        # Term 2-2 that appears in the tensor expansion of (a+b)^4
        term_1 = a.view(N, D, D, 1, 1) * b.view(N, 1, 1, D, D)
        term_2 = a.view(N, D, 1, D, 1) * b.view(N, 1, D, 1, D)
        term_3 = a.view(N, D, 1, 1, D) * b.view(N, 1, D, D, 1)
        term_4 = a.view(N, 1, D, D, 1) * b.view(N, D, 1, 1, D)
        term_5 = a.view(N, 1, D, 1, D) * b.view(N, D, 1, D, 1)
        term_6 = a.view(N, 1, 1, D, D) * b.view(N, D, D, 1, 1)
        res = term_1 + term_2 + term_3 + term_4 + term_5 + term_6
        assert res.shape == (N, D, D, D, D)
        return res

    else:
        raise ValueError(f"Invalid shapes: {a.shape}, {b.shape}")


@typecheck
def _point_moments(
    self,
    *,
    order: int = 2,
    features: Optional[Union[Float2dTensor, Double2dTensor]] = None,
    central: bool = False,
    rescale: bool = False,
    scale: Optional[Number] = None,
    dtype: Optional[Literal["float", "double"]] = None,
    **kwargs,
) -> Union[FloatTensor, DoubleTensor]:
    """Compute the local moments of a point cloud."""
    X = self.points if features is None else features

    if dtype == "float":
        X = X.float()
    elif dtype == "double":
        X = X.double()

    N = self.n_points
    D = X.shape[1]
    assert X.shape == (N, D)

    # We use a recursive formulation to best leverage our cache!
    def recursion(*, k: int):
        assert (k < order) or (central and k == order)
        mom = self.point_moments(
            order=k,
            features=features,
            central=False,
            rescale=False,
            scale=scale,
            dtype=dtype,
            **kwargs,
        )
        return mom

    # Thanks to caching, Conv will only be used if central=False
    if not central:
        Conv = self.point_convolution(
            scale=scale,
            normalize=True,
            dtype=dtype,
            cutoff=1e-4,
            **kwargs,
        )
        assert Conv.shape == (N, N)
    else:
        Conv = None

    if order == 1:
        if not central:
            moments = Conv @ X  # (N, D)
        else:
            # Use recursion (-> cache) to compute Conv @ X
            moments = recursion(k=1)
            # Centering an order 1 moment -> 0
            moments = moments - moments

        assert moments.shape == (N, D)

    elif order == 2:
        if not central:
            XX = X.view(N, D, 1) * X.view(N, 1, D)  # (N, D, D)
            moments = (Conv @ XX.view(N, D * D)).view(N, D, D)  # (N, D, D)
        else:
            # Use recursion (-> cache) to compute Conv @ X and Conv @ XX
            Xm = recursion(k=1)  # (N, D)
            moments = recursion(k=2)  # (N, D, D)
            moments = moments - (Xm.view(N, D, 1) * Xm.view(N, 1, D))  # (N, D, D)

        assert moments.shape == (N, D, D)

    elif order == 3:
        if not central:
            # X^3 as a (N, D, D, D) tensor
            XXX = X.view(N, D, 1, 1) * X.view(N, 1, D, 1) * X.view(N, 1, 1, D)
            moments = (Conv @ XXX.view(N, D * D * D)).view(N, D, D, D)  # (N, D, D, D)
        else:
            # Use recursion (-> cache) to compute Conv @ X, Conv @ XX, Conv @ X^3
            Xm = recursion(k=1)  # (N, D)
            mom_2 = recursion(k=2)  # (N, D, D)
            moments = recursion(k=3)  # (N, D, D, D)

            XmXmXm = Xm.view(N, D, 1, 1) * Xm.view(N, 1, D, 1) * Xm.view(N, 1, 1, D)

            moments = moments - symmetric_sum(Xm, mom_2) + 2 * XmXmXm

        assert moments.shape == (N, D, D, D)

    elif order == 4:
        if not central:
            # X^4 as a (N, D, D, D, D) tensor
            XXXX = (
                X.view(N, D, 1, 1, 1)
                * X.view(N, 1, D, 1, 1)
                * X.view(N, 1, 1, D, 1)
                * X.view(N, 1, 1, 1, D)
            )
            moments = (Conv @ XXXX.view(N, D * D * D * D)).view(N, D, D, D, D)
        else:
            # Use recursion (-> cache) for Conv @ X, Conv @ XX, Conv @ X^3, Conv @ X^4
            Xm = recursion(k=1)
            mom_2 = recursion(k=2)
            mom_3 = recursion(k=3)
            moments = recursion(k=4)

            XmXm = Xm.view(N, D, 1) * Xm.view(N, 1, D)
            XmXmXmXm = (
                Xm.view(N, D, 1, 1, 1)
                * Xm.view(N, 1, D, 1, 1)
                * Xm.view(N, 1, 1, D, 1)
                * Xm.view(N, 1, 1, 1, D)
            )

            moments = (
                moments
                - symmetric_sum(Xm, mom_3)
                + symmetric_sum(XmXm, mom_2)
                - 3 * XmXmXmXm
            )

        assert moments.shape == (N, D, D, D, D)

    else:
        raise NotImplementedError(
            f"Moments of order {order} have not been implemented."
        )

    if rescale:
        if scale is None:
            raise ValueError("A finite scale must be provided if rescale is True")

        moments = moments / scale**order

    if dtype == "float":
        assert moments.dtype == torch.float32
    elif dtype == "double":
        assert moments.dtype == torch.float64

    return moments
