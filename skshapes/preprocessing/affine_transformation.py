# TODO : write affine operation with dim_mat = dim_shape + 1
# TODO : input of AffineTransformation must be a list of matrices

import torch
import numpy as np
from ..data import Dataset, Shape


class AffineTransformation:
    def __init__(self, matrix):
        # Check that the matrix is a square matrix and convert it to torch if necessary
        if isinstance(matrix, list):
            if len(set([len(l) for l in matrix])) != 1:
                raise ValueError("The matrix must be a square matrix")
            else:
                self.matrix = torch.tensor(matrix, dtype=torch.float32)

        if isinstance(matrix, np.ndarray):
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("The matrix must be a square matrix")
            else:
                self.matrix = torch.from_numpy(matrix).float()

        elif isinstance(matrix, torch.Tensor):
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("The matrix must be a square matrix")
            else:
                self.matrix = matrix

    def fit(self, shapes):
        dim_mat = self.matrix.shape[0]
        dim_shapes = shapes[0].dim
        # Check that the dimension of the matrix is dim of the shape or dim of the shape + 1
        if dim_mat not in [dim_shapes, dim_shapes + 1]:
            raise ValueError(
                "The dimension of the matrix must be the same as the dimension of the shapes or the dimension of the shapes + 1"
            )

    def transform(self, shapes):
        dim_mat = self.matrix.shape[0]
        dim_shapes = shapes[0].dim

        if dim_mat == dim_shapes:
            new_points = [(self.matrix @ shape.points.T).T for shape in shapes]

        if isinstance(shapes, Dataset):
            for i, shape in enumerate(shapes.shapes):
                setattr(shape, "points", new_points[i])
                return shapes
        else:
            for i, shape in enumerate(shapes.shapes):
                setattr(shape, "points", new_points[i])
                return Dataset(shapes=shapes)

    def fit_transform(self, shapes):
        self.fit(shapes)
        return self.transform(shapes)
