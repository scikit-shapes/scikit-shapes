from ..types import (
    typecheck,
    FloatTensor,
    Optional,
    Float2dTensor,
    Number,
    Literal,
)


@typecheck
def symmetric_sum(a: FloatTensor, b: FloatTensor) -> FloatTensor:
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
def point_moments(
    self,
    *,
    order: int = 2,
    features: Optional[Float2dTensor] = None,
    central: bool = False,
    rescale: bool = False,
    scale: Optional[Number] = None,
    **kwargs,
) -> FloatTensor:
    """Compute the local moments of a point cloud."""
    X = self.points if features is None else features

    N = self.n_points
    D = X.shape[1]
    assert X.shape == (N, D)

    Conv = self.point_convolution(scale=scale, normalize=True, **kwargs)
    assert Conv.shape == (N, N)

    if order == 1:
        moments = Conv @ X  # (N, D)

        if central:
            # Centering an order 1 moment -> 0
            moments = moments - moments

        assert moments.shape == (N, D)

    elif order == 2:
        XX = X.view(N, D, 1) * X.view(N, 1, D)  # (N, D, D)
        moments = (Conv @ XX.view(N, D * D)).view(N, D, D)  # (N, D, D)

        if central:
            # (N, D)
            Xm = self.point_moments(order=1, features=features, scale=scale, **kwargs)
            moments = moments - (Xm.view(N, D, 1) * Xm.view(N, 1, D))  # (N, D, D)

        assert moments.shape == (N, D, D)

    elif order == 3:
        # X^3 as a (N, D, D, D) tensor
        XXX = X.view(N, D, 1, 1) * X.view(N, 1, D, 1) * X.view(N, 1, 1, D)
        moments = (Conv @ XXX.view(N, D * D * D)).view(N, D, D, D)  # (N, D, D, D)

        if central:
            Xm = self.point_moments(order=1, features=features, scale=scale, **kwargs)
            XmXmXm = Xm.view(N, D, 1, 1) * Xm.view(N, 1, D, 1) * Xm.view(N, 1, 1, D)

            # (N, D, D)
            mom_2 = self.point_moments(
                order=2, features=features, scale=scale, **kwargs
            )

            moments = moments - symmetric_sum(Xm, mom_2) + 2 * XmXmXm

        assert moments.shape == (N, D, D, D)

    elif order == 4:
        # X^4 as a (N, D, D, D, D) tensor
        XXXX = (
            X.view(N, D, 1, 1, 1)
            * X.view(N, 1, D, 1, 1)
            * X.view(N, 1, 1, D, 1)
            * X.view(N, 1, 1, 1, D)
        )
        moments = (Conv @ XXXX.view(N, D * D * D * D)).view(N, D, D, D, D)

        if central:
            Xm = self.point_moments(order=1, features=features, scale=scale, **kwargs)
            XmXm = Xm.view(N, D, 1) * Xm.view(N, 1, D)
            XmXmXmXm = (
                Xm.view(N, D, 1, 1, 1)
                * Xm.view(N, 1, D, 1, 1)
                * Xm.view(N, 1, 1, D, 1)
                * Xm.view(N, 1, 1, 1, D)
            )

            mom_2 = self.point_moments(
                order=2, features=features, scale=scale, **kwargs
            )
            mom_3 = self.point_moments(
                order=3, features=features, scale=scale, **kwargs
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

    return moments
