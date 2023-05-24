import torch


class LandmarkLoss:
    def __init__(self, p=2) -> None:
        self.p = p

    def fit(self, *, source, target):
        self.source_landmarks = source.landmarks
        self.target_landmarks_points = target.points[target.landmarks]

    def __call__(self, x):
        """
        Args:
            x (torch.Tensor): the current mesh
        Returns:
            loss (torch.Tensor): the data attachment loss
        """
        return torch.norm(
            x[self.source_landmarks] - self.target_landmarks_points, p=self.p
        )
