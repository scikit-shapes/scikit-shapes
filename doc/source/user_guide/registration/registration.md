Registration
============

Presentation
------------

Registration is the task of finding a suitable transformation from a source to a target shape.

A registration task must be at least parametrized with a `deformation model` and a `loss function`

- The deformation model specifies constrains about the way source can be transformed to match target.
- The loss function measure the discrepancy between the morphed source and the target

```python
import skshapes as sks

# Source and target are circles, the difference between these is a translation
source = sks.Circle()
target = sks.Circle()
target.points += torch.tensor([1.0, 2.0], dtype=sks.float_dtype)
# Define loss and deformation model
loss = sks.L2Loss()
model = sks.RigidMotion()
# Initialize the registration object
r = sks.Registration(
    model=model,
    loss=loss,
)
# Fit the registration
r.fit(
    source=source,
    target=target,
)
# Print the translation parameter
print(r.translation_)
```
```
tensor([1., 2.])
```

Choosing a Loss function
------------------------

A loss function is a way to quantify the difference between two shapes. In scikit-shapes a loss function is represented by a class that can be initialized with some hyperparameters
```python
import skshapes as sks

l1_loss = sks.LpLoss(p=1)
```
Linear combination of loss function are valid loss functions:
```python
import skshapes as sks

custom_loss = 2 * sks.LandmarkLoss() + sks.NearestNeighborsLoss()
```
Some losses requires that `source` and `target` fulfill certains conditions:

- for polydatas

| Loss function          | Description                          | Restrictions                                            |
| ---------------------- | ------------------------------------ | ------------------------------------------------------- |
| `LpLoss`               | Lp loss for vertices                 | `source` and `target` must be in correspondence         |
| `L2Loss`               | L2 loss for vertices                 | `source` and `target` must be in correspondence         |
| `LandmarkLoss`         | Lp loss for landmarks                | `source` and `target` must have corresponding landmarks |
| `NearestNeighborsLoss` | Nearest neighbors distance           | NA                                                      |

- for images

| Loss function          | Description                          | Restrictions                                            |
| ---------------------- | ------------------------------------ | ------------------------------------------------------- |



Choosing a Registration model
-----------------------------

| Deformation model      | Description
| ---------------------- | ------------------------------------------------- |
| `RigidMotion`          | Rotation + translation                            |
| `AffineDeformation`    | Affine transformation                             |
| `IntrinsicDeformation` | Sequence of                                       |
| `ExtrinsicDeformation` | Distord the ambiant space to make the shape move  |
