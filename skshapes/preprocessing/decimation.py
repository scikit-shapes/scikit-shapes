# TODO : write decimation operation with landmarks == "all" (parallel decimation)
import torch
from ..data import Dataset, Shape


class Decimation:
    """A transformer to apply decimation on a list of shapes or a Dataset object,
     it can be initialized with the target reduction (between 0 and 1) and it will
     decimate each shape to the target reduction. Note that is it is applied on a Dataset
     with landmarks = "all", the decimation must occur in parallel to keep the correspondence.

    The fit_transform method returns a Dataset object with the decimated shapes
    """

    def __init__(self, target_reduction):
        """Initialize the Decimation object with the target reduction

        Args:
            target_reduction (float): the target reduction between 0 and 1
        """
        self.target_reduction = target_reduction

    def fit(self, shapes):
        """The fit method does nothing in this case

        Args:
            shapes (list of Shape or Dataset): the shapes on which the decimation will be applied
        """
        pass

    def transform(self, shapes):
        """Apply the decimation on the shapes

        Args:
            shapes (list of Shape or Dataset): the shapes on which the decimation will be applied

        Returns:
            Dataset (skshapes.data.Dataset): the dataset with decimated shapes
        """

        # If the points are not in correspondence, we simply decimate each shape independently
        if not isinstance(shapes, Dataset) or shapes.landmarks != "all":

            # In this case, we simply decimate independently each shape
            new_polydata = [
                shape.to_pyvista().decimate_pro(
                    self.target_reduction, preserve_topology=True
                )
                for shape in shapes.shapes
            ]
            new_shapes = [Shape.from_pyvista(polydata) for polydata in new_polydata]

        else:  # ie shapes is a Dataset and landmarks are "all"
            # TODO : write parallel decimation
            # NotImplelented
            raise NotImplementedError(
                "Decimation is not implemented yet in the case of landmarks = 'all'"
            )

        # Return a dataset with the new shapes
        if isinstance(shapes, Dataset):
            setattr(shapes, "shapes", new_shapes)
            return shapes

        else:
            return Dataset(shapes=shapes)

    def fit_transform(self, shapes):
        """Apply the decimation on the shapes

        Args:
            shapes (list of Shape or Dataset): the shapes on which the decimation will be applied

        Returns:
            Dataset (skshapes.data.Dataset): the dataset with decimated shapes
        """
        self.fit(shapes)
        return self.transform(shapes)
