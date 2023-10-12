from .elastic_metric import ElasticMetric
from .splines import SplineDeformation
from .rigid import RigidMotion

from typing import Union

Model = Union[ElasticMetric, RigidMotion, SplineDeformation]
