"""Tests related to the device compatibility of shapes."""

from typing import get_args

import pytest
import pyvista.examples
import torch

import skshapes as sks
from skshapes.errors import DeviceError

shape1 = sks.PolyData(pyvista.Sphere())
shape2 = sks.PolyData(pyvista.Sphere()).resample(ratio=0.5)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Cuda is required for this test"
)
def test_loss_device():
    """Error if source and target are on different devices."""
    source = shape1.to("cpu")
    target = shape2.to("cuda")

    list_of_losses = [
        i
        for i in sks.loss.__dict__.values()
        if isinstance(i, type) and i in get_args(sks.Loss)
    ]
    for loss in list_of_losses:
        loss_fn = loss()

        with pytest.raises(DeviceError):
            loss_fn(source=source, target=target)
