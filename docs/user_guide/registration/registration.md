# Registration

## Presentation

Registration is the task of finding a suitable transformation from a source to a target shape.

A registration task must be at least parametrized with a `deformation model` and a `loss function`

- The deformation model specifies contrains about the way source can be transformed to match target.
- The loss function measure the discrepency between the morphed source and the target

```python
import skshapes as sks

# Source and target are circles, the difference between these is a translation
source = sks.Circle()
target = sks.Circle()
target.points += torch.tensor([1.0, 2.0], dtype=sks.float_dtype)
# Define loss and deformation model
loss = sks.L2Loss()
model = sks.RigidMotion()
# Initialize the registration object
r = sks.Registration(
    model=model,
    loss=loss,
)
# Fit the registration
r.fit(
    source=source,
    target=target,
)
# Print the translation parameter
print(r.translation_)
```
```
tensor([1., 2.])
```

## Choosing a Loss function

Some losses requires that `source` and `target` fulfill certains conditions

- for polydatas

| Loss function          | Description                          | Restrictions                                            |
| ---------------------- | ------------------------------------ | ------------------------------------------------------- |
| `L2Loss`               | L2 loss for vertices                 | `source` and `target` must be in correspondance         |
| `LandmarkLoss`         | L2 loss for landmarks                | `source` and `target` must have corresponding landmarks |
| `NearestNeighborsLoss` | Nearest neighbors distance           | NA                                                      |

- for images 

| Loss function          | Description                          | Restrictions                                            |
| ---------------------- | ------------------------------------ | ------------------------------------------------------- |

## Choosing a Registration model

| Deformation model      | Description
| ---------------------- | ------------------------------------------------- |
| `RigidMotion`          | Rotation + translation                            |
| `AffineDeformation`    | Affine transformation                             |
| `IntrinsicDeformation` | Sequence of                                       |
| `ExtrinsicDeformation` | Distord the ambiant space to make the shape move  |
