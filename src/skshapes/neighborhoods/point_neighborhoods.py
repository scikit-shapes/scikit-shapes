from typing import Literal

from ..input_validation import typecheck
from .graph_neighborhoods import GraphNeighborhoods
from .kernel_neighborhoods import KernelNeighborhoods
from .trivial_neighborhoods import TrivialNeighborhoods


@typecheck
def _point_neighborhoods(
    self,
    *,
    radius: float | None = None,
    n_neighbors: int | None = None,
    window: Literal["gaussian", "ball", "knn"] = "gaussian",
    distance: Literal["ambient", "geodesic"] = "ambient",
):

    if window == "knn":
        if n_neighbors is None:
            msg = "When window is 'knn', n_neighbors must be provided."
            raise ValueError(msg)

        if distance == "ambient":
            return GraphNeighborhoods(
                graph=self.knn_graph(n_neighbors=n_neighbors)
            )
        else:
            msg = "Geodesic distances are not supported yet."
            raise NotImplementedError(msg)
    elif radius is None:
        return TrivialNeighborhoods()
    elif distance == "ambient":
        return KernelNeighborhoods(
            points=self.points,
            window=window,
            radius=radius,
        )
    else:
        msg = "Geodesic distances are not supported yet."
        raise NotImplementedError(msg)
