# Transformer :
# __init__(hyperparameters)
# fit(X)
# transform(X)
# fit_transform(X, y=None)


# TODO : write affine operation with dim_mat = dim_shape + 1

import numpy as np
import torch
from ..data import Dataset, Shape

from .landmarks_setter import LandmarkSetter
from .decimation import Decimation
from .affine_transformation import AffineTransformation


class Pipeline:
    def __init__(self, steps):
        self.steps = steps

    def fit_transform(self, shapes):
        dataset = shapes
        for step in self.steps:
            dataset = step.fit_transform(shapes=dataset)

        return dataset
