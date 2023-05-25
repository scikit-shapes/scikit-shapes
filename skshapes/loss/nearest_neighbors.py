from pykeops.torch import Vi, Vj

import torch


class NearestNeighborsLoss:
    def __init__(self):
        pass

    def fit(self, *, source, target):
        self.target_points = target.points

    def __call__(self, x):
        """
        Args:
            x (torch.Tensor): the current mesh
        Returns:
            loss (torch.Tensor): the data attachment loss
        """
        X_i = Vi(x)
        X_j = Vj(self.target_points)

        D_ij = ((X_i - X_j) ** 2).sum(-1)
        correspondences = D_ij.argKmin(K=1, dim=1).view(-1)

        return torch.norm((x - self.target_points[correspondences]), p=2, dim=1).mean()
