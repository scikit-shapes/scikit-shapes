from typing import Literal

from ..input_validation import typecheck
from ..types import Number
from .graph_neighborhoods import GraphNeighborhoods
from .kernel_neighborhoods import KernelNeighborhoods
from .trivial_neighborhoods import TrivialNeighborhoods


@typecheck
def _point_neighborhoods(
    self,
    *,
    method: Literal[
        "auto",
        "trivial",
        "gaussian kernel",
        "ball kernel",
        "knn graph",
        "cotan laplacian",
        "intrinsic cotan laplacian",
        "tufted laplacian",
    ] = "auto",
    scale: Number | None = None,
    n_neighbors: int | None = None,
    n_normalization_iterations: int | None = None,
    smoothing_method: Literal[
        "auto", "exact", "exp(x)=1/(1-x)", "nystroem"
    ] = "auto",
    laplacian_method: Literal["auto", "exact", "log(x)=x-1"] = "auto",
):

    if method == "auto":
        # For triangle meshes, we may consider the tufted Laplacian instead.
        method = "trivial" if scale is None else "gaussian kernel"

    if method == "trivial":
        return TrivialNeighborhoods(
            masses=self.point_masses,
        )

    elif method == "knn graph":
        if n_neighbors is None:
            msg = 'When method is "knn graph", n_neighbors must be provided.'
            raise ValueError(msg)

        return GraphNeighborhoods(
            graph=self.knn_graph(n_neighbors=n_neighbors),
            masses=self.point_masses,
            n_normalization_iterations=n_normalization_iterations,
            smoothing_method=smoothing_method,
            laplacian_method=laplacian_method,
        )

    elif method in ["gaussian kernel", "ball kernel"]:
        if scale is None:
            msg = f'When method is "{method}", scale must be provided.'
            raise ValueError(msg)

        return KernelNeighborhoods(
            points=self.points,
            window=method,
            scale=scale,
            masses=self.point_masses,
            n_normalization_iterations=n_normalization_iterations,
            smoothing_method=smoothing_method,
            laplacian_method=laplacian_method,
        )
    else:
        msg = f'Point neighborhoods with the "{method}" method are not supported yet.'
        raise NotImplementedError(msg)
