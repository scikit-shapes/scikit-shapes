from .elastic_metric import ElasticMetric
from .kernel import KernelDeformation
from .rigid import RigidMotion

from beartype.typing import Union

Model = Union[ElasticMetric, RigidMotion, KernelDeformation]
