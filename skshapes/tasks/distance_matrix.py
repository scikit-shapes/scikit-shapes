import torch
import numpy as np
from .registration import Registration


class DistanceMatrix:
    def __init__(
        self,
        *,
        model,
        loss,
        optimizer="LBFGS",
        n_iter=10,
        verbose=0,
        n_steps=1,
        regularization=1,
        device="auto",
        **kwargs,
    ) -> None:

        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.verbose = verbose

        self.registration = Registration(
            model=model,
            loss=loss,
            optimizer=optimizer,
            n_iter=n_iter,
            verbose=verbose,
            n_steps=n_steps,
            regularization=regularization,
            device=device,
        )

    def fit(
        self,
        *,
        shapes,
    ) -> None:

        # Make copies of the shapes and move them to the device
        shapes = [shape.copy().to(self.device) for shape in shapes]
        n = len(shapes)
        self.distance_matrix = np.zeros((n, n))

        # Load the tensors and fit the models/loss
        # Loop over pairs of shapes
        for i in range(len(shapes)):
            for j in range(i + 1, len(shapes)):

                if self.verbose > 0:
                    print(f"Fitting shapes {i} and {j}...")

                self.registration.fit(source=shapes[i], target=shapes[j])
                self.distance_matrix[i, j] = self.registration.distance
                self.distance_matrix[j, i] = self.distance_matrix[i, j]

    def fit_transform(
        self,
        *,
        shapes,
    ) -> np.ndarray:

        self.fit(shapes=shapes)
        return self.distance_matrix