## Writing a deformation model

A deformation model specifies the type of deformation to apply to a shape. Examples include

- `Rigidmotion`
- `ExtrinsicDeformation`
- `IntrinsicDeformation`

This document is intended to explain how deformation models must be implemented to be integrated into scikit-shapes pipelines.

## Structure

A deformation model must inherits from `sks.morphing.BaseModel` and following methods are required :

- `__init__` : class constructor used to define some hyperparameters
- `morph` : method to apply the deformation to the shape given a parameter. The annotations are imposed in the tests

```python
def morph(
    self,
    shape: shape_type,
    parameter: torch.Tensor,
    return_path: bool,
    return_regularization: bool
    ) -> MorphingOutput
```
- `parameter_shape(shape=...)`: method that output a tuple corresponding to the shape of the parameter that must be passed to `morph`
- `initial_parameter(shape=...)`: method that returns a default parameter. For now it is a zero `torch.tensor` defined on the right device and with the right shape.

## Usage

Deformation objects are accessed by two ways : by the end-user of scikit-shapes and internally by a task object (such as `sks.Registration()`).


End-user interacts with `deformation`:

- at initialization, by setting hyperparameters
- inderectly by accessing MorphingOuput after fitting `task`

Task interact with `deformation`:

- by calling `initial_parameter`
- by calling `morph(shape, parameter)`
- by copying the content of `MorphingOutput`

Tasks relies on optimization schemes and `morph` must be compatible with `autograd` the following test must tun without error (it is actually in the test suite) 

```python

source, target = ... # Any shapes
loss = ... # Any loss
# Initialize the deformation model
model = deformation_model()
# Get an initial parameter
p = model.inital_parameter(shape=source)
p.requires_grad_(True)

morphed_shape = model.morph(shape=source, parameter=p).morphed_shape
L = loss(morphed_shape, target)
L.backward()
assert p.grad is not None

if torch.cuda.is_available():
    p = p.cuda()
    try:
        model.morph(shape=source, parameter=p).morphed_shape
    except ValueError as e:
        pass
    else:
        raise RuntimeError("Expected ValueError as parameter and shape not on the same device")
```


## Access to shape properties

Some deformation models require some attributes to be available for morphed shape. An example is `shape.control_points` in `ExtrinsicDeformation()`. These attributes should not be added to the arguments of `morph`, but instead accessed inside `.morph()` with a clear error message if the attribute cannot be reached and if no default behavior can be defined in this case.


## Custom output attributes

The output of `.morph()` is a `MorphingOutput`. It is simply a class with no method and containing by default the following attributes :

- `morphed_shape` : the shape morphed by the model
- `path` : a list of shapes, from `source` to `morphed_shape` of length `n_steps + 1`
- `regularization` : the regularization corresponding to the model and the parameter

Other attributes can be written in `.morph()`, for example for `RigidMotion`, you can access `.rotation_matrix()` or `.translation()`

After fitting, all the attributes of the fitted deformation model are accessible in the registration with the same name as in `MorphingOutput` plus an extra "_":

```python
model = RigidMotion()
r = Registration(model=model, los=..., ...)
r.fit(source=source, target=target)

rotation_matrix = r.rotation_matrix_
translation = r.translatio_
output = r.morphed_shape_
```
