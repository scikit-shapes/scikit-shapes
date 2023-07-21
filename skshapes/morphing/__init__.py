from .elastic_metric import ElasticMetric
from .rigid import RigidMotion

from typing import Union

Model = Union[ElasticMetric, RigidMotion]
