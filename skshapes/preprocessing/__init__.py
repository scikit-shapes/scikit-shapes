# Transformer :
# __init__(hyperparameters)
# fit(X)
# transform(X)
# fit_transform(X, y=None)

# TODO : write decimation operation with landmarks different from None
# TODO : write affine operation with dim_mat = dim_shape + 1

import numpy as np
import torch
from ..data import Dataset, Shape


class Pipeline:

    def __init__(self, steps):

        self.steps = steps

    def fit_transform(self, shapes):

        dataset = shapes
        for step in self.steps:
            dataset = step.fit_transform(shapes=dataset)

        return dataset

class LandmarkSetter:

    def __init__(self, landmarks):
        # Landmarks can be :
        # - a list of list of integers with same length
        # - the keyword "all" to set all the points as landmarks

        if landmarks == "all":
            pass

        #If landmarks is a list (and not a string)
        elif isinstance(landmarks, list) and not isinstance(landmarks, str):
            # Check that the number of landmarks is the same for each shape
            if len(set([len(l) for l in landmarks])) != 1:
                raise ValueError("The number of landmarks must be the same for each shape")
        
        else:
            raise ValueError("Landmarks must be a list of list of integers or the keyword 'all'")
                             
        self.landmarks = landmarks

    def fit(self, shapes):

        if self.landmarks != "all":
            # Check that the number of shapes is the same as the number of landmarks
            if len(shapes) != len(self.landmarks):
                raise ValueError("The number of shapes and the number of landmarks must be the same")
            
            # Check that the landmarks are in the shape
            for shape, l in zip(shapes, self.landmarks):
                if np.max(l) >= shape.points.shape[0]:
                    raise ValueError("Landmarks {} cannot be associated to shape of size {}".format(np.max(l), shape.points.shape[0]))
            
    def fit_transform(self, shapes):
        self.fit(shapes)

        #Check wether shape is a list or a Dataset
        if isinstance(shapes, Dataset):

            setattr(shapes, "landmarks", self.landmarks)
            return shapes
        
        else:
            return Dataset(shapes=shapes, landmarks=self.landmarks)


class Decimation:

    def __init__(self, target_reduction):
        self.target_reduction = target_reduction

    def fit(self, shapes):
        pass

    def fit_transform(self, shapes):
        self.fit(shapes)

        #Check wether shape is a list or a Dataset
        if isinstance(shapes, Dataset):
            
            if shapes.landmarks is None:
                #In this case, we simply decimate independently each shape
                new_polydata = [shape.to_pyvista().decimate_pro(self.target_reduction, preserve_topology=True) for shape in shapes.shapes]
                new_shapes = [Shape.from_pyvista(polydata) for polydata in new_polydata]

                setattr(shapes, "shapes", new_shapes)
                return shapes

            elif shapes.landmarks != "all":
                # In this case, a list of landmarks is given and we must ensure that the decimation does not change the landmarks
                landmarks_points = torch.cat([shapes.shape[i].points[shapes.landmarks[i]] for i in range(len(shapes))])
                print(landmarks_points)

            else:
                # In the last scenario, all the points are landmarks and we can simply decimate one shape and extract the
                # same points as landmarks for all the shapes
                pass

        else:
            # If we provide a list of shapes, we simply decimate each shape independently (as if landmarks were None)
            new_polydata = [shape.to_pyvista().decimate_pro(self.target_reduction, preserve_topology=True) for shape in shapes]
            new_shapes = [Shape.from_pyvista(polydata) for polydata in new_polydata]

            return Dataset(shapes=new_shapes)
        
class AffineTransformation:

    def __init__(self, matrix):
        # Check that the matrix is a square matrix and convert it to torch if necessary
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
            raise ValueError("The dimension of the matrix must be the same as the dimension of the shapes or the dimension of the shapes + 1")
        
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


            
