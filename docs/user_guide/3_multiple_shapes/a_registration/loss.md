## Loss functions

Losses function represent discrepancy between two shapes. In scikit-shapes, there are represented by classes that must fulfills certain condition to be integrable in registration or other tasks.
This document presents how a loss class is structured, it is useful if you intend to write custom losses.

## Structure

Losses must inherit from `sks.BaseLoss` and must contain two methods

- `__init__()` which can be called with or without arguments
- `__call__(source, target)` which must be called with a pair of shapes


As an example, there is an implementation of the L1 loss (it is already available with `sks.LpLoss(p=1)`):
```python
import skshapes as sks


class L1Loss(sks.BaseLoss):
    def __init__(self):
        pass

    def __call__(self, source, target):
        return torch.abs(source.points - target.points).sum(dim=-1).mean()
```

Note that the `__call__` method must be compatible with pytorch autograd: the gradient with respect to `source.points` needs to be computed for optimization purposes.

## Access to shape properties

Some loss functions require some attributes to be available for shapes. An example is `shape.landmarks` in `LandmarkLoss()`. These attributes should not be added to the arguments of `__call__`, but instead accessed inside `__call__` with a clear error message if the attribute cannot be reached and if no default behavior can be defined in this case.


## Indication about resctriction for currently implemented losses

- for polydatas

| Loss function          | Description                          | Restrictions                                            |
| ---------------------- | ------------------------------------ | ------------------------------------------------------- |
| `L2Loss`               | L2 loss for vertices                 | `source` and `target` must be in correspondence         |
| `LandmarkLoss`         | L2 loss for landmarks                | `source` and `target` must have corresponding landmarks |
| `NearestNeighborsLoss` | Nearest neighbors distance           | NA                                                      |

- for images

| Loss function          | Description                          | Restrictions                                            |
| ---------------------- | ------------------------------------ | ------------------------------------------------------- |
