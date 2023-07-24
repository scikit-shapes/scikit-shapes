from ..types import (
    typecheck,
    FloatTensor,
    Optional,
    Float2dTensor,
    Number,
    Literal,
)


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
        XXT = X.view(N, D, 1) * X.view(N, 1, D)  # (N, D, D)
        moments = (Conv @ XXT.view(N, D * D)).view(N, D, D)  # (N, D, D)

        if central:
            Xm = Conv @ X  # (N, D)
            moments = moments - (Xm.view(N, D, 1) * Xm.view(N, 1, D))  # (N, D, D)

        assert moments.shape == (N, D, D)

    else:
        raise NotImplementedError(
            f"Moments of order {order} have not been implemented."
        )

    if rescale:
        if scale is None:
            raise ValueError("A finite scale must be provided if rescale is True")

        moments = moments / scale**order

    return moments
