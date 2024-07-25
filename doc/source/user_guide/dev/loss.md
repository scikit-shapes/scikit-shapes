Loss functions
==============

Losses function represent discrepancy between two shapes

Structure
---------

Losses contain two methods

- `__init__()` which can be called with or without arguments
- `__call__()` which must be called with a pair of shapes

Access to shape properties
--------------------------

Some loss functions require some attributes to be available for shapes. An example is `shape.landmarks` in `LandmarkLoss()`. These attributes should not be added to the arguments of `__call__`, but instead accessed inside `__call__` with a clear error message if the attribute cannot be reached and if no default behavior can be defined in this case.


Indication about restriction for actually implemented losses
------------------------------------------------------------

- for polydatas

| Loss function          | Description                          | Restrictions                                            |
| ---------------------- | ------------------------------------ | ------------------------------------------------------- |
| `L2Loss`               | L2 loss for vertices                 | `source` and `target` must be in correspondence         |
| `LandmarkLoss`         | L2 loss for landmarks                | `source` and `target` must have corresponding landmarks |
| `NearestNeighborsLoss` | Nearest neighbors distance           | NA                                                      |

- for images

| Loss function          | Description                          | Restrictions                                            |
| ---------------------- | ------------------------------------ | ------------------------------------------------------- |
