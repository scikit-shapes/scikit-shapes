import torch
from ..data import Dataset, Shape


class LandmarkSetter:
    """A transformer to set landmarks on a list of shapes or a Dataset object
    it can be initialized with the keyword "all" to set all the points as landmarks
    or with :
    - a list of list of integers with same length (indices of the landmarks for each shape)
    - a list of point clouds

    The fit_transform method returns a Dataset object with the landmarks set
    """

    def __init__(self, landmarks, by_indices=False):
        """Initialize the LandmarkSetter object with the landmarks, which can be :
        - 'all' if it is intendend to be applied on shapes that are in correspondence
        - if by_indices = True : a list of list of integers with same length (indices of the landmarks for each shape)
        - if by_indices = False : a list of point clouds
        TODO : is it useful to have the by_indices parameter ? Maybe it is better to guess the type of landmarks

        The validity of the landmarks is checked.

        Args:
            landmarks (_type_): _description_
            by_indices (bool, optional): True if the landmarks indices are passed, False if landmarks are passed as point clouds. Defaults to False.
        """

        # 1) Check that landmarks is a string or a valid object with __getitem__ method and store the type
        if landmarks == "all":
            self.landmarks_type = "all"

        elif by_indices:
            self.landmarks_type = "list_of_list_of_indices"
            # Check that the number of landmarks is the same for each shape
            if len(set([len(l) for l in landmarks])) != 1:
                raise ValueError(
                    "The number of landmarks must be the same for each shape"
                )

        else:
            self.landmarks_type = "list_of_point_clouds"

            # Check that the shapes are all of the same size
            if (
                len(set([l.shape[0] for l in landmarks])) != 1
                and len(set([l.shape[1] for l in landmarks])) != 1
            ):
                raise ValueError("All landmarks must have the same size")

        # Store the landmarks
        self.landmarks = landmarks

    def fit(self, shapes):
        """Check that the landmarks are valid

        Args:
            shapes (list of Shape or Dataset): the shapes on which the landmarks will be set
        """

        if self.landmarks_type == "all":
            # Check that the shapes are all of the same size
            if len(set([shape.points.shape[0] for shape in shapes])) != 1:
                raise ValueError(
                    "All shapes must have the same size to set all points as landmarks"
                )

        elif self.landmarks_type == "list_of_list_of_indices":
            self.landmarks = [
                shape.points[landmarks]
                for shape, landmarks in zip(shapes, self.landmarks)
            ]

        elif self.landmarks_type == "list_of_point_clouds":
            # convert landmarks to a list of torch.Tensor if necessary
            self.landmarks = [
                torch.Tensor(l) if not isinstance(l, torch.Tensor) else l
                for l in self.landmarks
            ]

    def transform(self, shapes):
        """Return a Dataset object with the landmarks set

        Args:
            shapes (list of Shape or Dataset): the shapes on which the landmarks will be set

        Returns:
            Dataset (skshapes.data.Dataset): the dataset with the landmarks set
        """

        # Check wether shape is a list or a Dataset
        if isinstance(shapes, Dataset):

            setattr(shapes, "landmarks", self.landmarks)
            return shapes

        else:
            return Dataset(shapes=shapes, landmarks=self.landmarks)

    def fit_transform(self, shapes):
        """Check that the landmarks are valid and return a Dataset object with the landmarks set

        Args:
            shapes (list of Shape or Dataset): the shapes on which the landmarks will be set

        Returns:
            Dataset (skshapes.data.Dataset): the dataset with the landmarks set
        """
        self.fit(shapes)
        return self.transform(shapes)
