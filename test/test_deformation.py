import skshapes as sks
import torch

deformation_models = [
    sks.IntrinsicDeformation,
    sks.ExtrinsicDeformation,
    sks.RigidMotion,
]


def test_deformation():
    for deformation_model in deformation_models:
        _test(deformation_model)


def _test(deformation_model):
    # Define a pair of shapes and a loss function
    shape = sks.Sphere().decimate(target_reduction=0.99)
    target = shape.copy()
    target.points += 1
    loss = sks.L2Loss()

    # Initialize the deformation model
    model = deformation_model()
    # Get an initial parameter
    p = model.inital_parameter(shape=shape)

    p.requires_grad_(True)

    morphed_shape = model.morph(shape=shape, parameter=p).morphed_shape
    L = loss(morphed_shape, target)

    L.backward()
    assert p.grad is not None

    if torch.cuda.is_available():
        p = p.cuda()
        try:
            model.morph(shape=shape, parameter=p).morphed_shape
        except ValueError as e:
            pass
        else:
            raise RuntimeError("Expected ValueError")
