from .optimal_transport import OptimalTransportLoss
from .landmarks import LandmarkLoss
from .nearest_neighbors import NearestNeighborsLoss
from .l2 import L2Loss
from .baseloss import EmptyLoss, SumLoss, ProductLoss


from typing import Union

Loss = Union[
    OptimalTransportLoss,
    LandmarkLoss,
    NearestNeighborsLoss,
    L2Loss,
    EmptyLoss,
    SumLoss,
    ProductLoss,
]
