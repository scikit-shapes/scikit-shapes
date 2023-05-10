# Transformer :
# __init__(hyperparameters)
# fit(X)
# transform(X)
# fit_transform(X, y=None)

import numpy as np
import torch
from ..data import Dataset, Shape

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


# class Decimation:

#     def __init__(self, target_reduction):
#         self.target_reduction = target_reduction

#     def fit(self, shapes):
#         pass

#     def fit_transform(self, shapes):
#         self.fit(shapes)

#         #Check wether shape is a list or a Dataset
#         if isinstance(shapes, Dataset):
            
#             if shapes.landmarks is None:
#                 pass
#             elif shapes.landmarks == "all":
#                 pass
#             else:
#                 landmarks_points = torch.cat([shapes.shape[i].points[shapes.landmarks[i]] for i in range(len(shapes))])

            
#             if shapes.landmarks is not None:

#                 landmarks_points = torch.cat([shapes.shape[i].points[shapes.landmarks[i]] for i in range(len(shapes))])

#             new_polydata = [shape.to_pyvista().decimate_pro(self.target_reduction, preserve_topology=True) for shape in shapes.shapes]
#             new_shapes = [Shape.from_pyvista(polydata) for polydata in new_polydata]

#             shapes.shapes = [s.to_decimate(self.factor) for s in shapes.shapes]
#             return shapes
        

#         else:


#             return Dataset(shapes=[s.decimate(self.factor) for s in shapes])

            
