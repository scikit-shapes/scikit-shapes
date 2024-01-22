"""Converters for arguments."""

from typing import Union
from functools import wraps
import torch
import numpy as np
from ..types import float_dtype, int_dtype


def _convert_arg(x: Union[np.ndarray, torch.Tensor]):
    """Convert an array to the right type.

    Depending on the type of the input, it converts the input to the right
    type (torch.Tensor) and convert the dtype of the tensor to the right one
    (float32 for float, int64 for int).

    Parameters
    ----------
    x : Union[np.ndarray, torch.Tensor])
        the input array

    Raises
    ------
    ValueError
        if the input is a complex tensor

    Returns
    -------
    torch.Tensor
        corresponding tensor with the right dtype

    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    if isinstance(x, torch.Tensor):
        if torch.is_floating_point(x) and x.dtype != float_dtype:
            return x.to(float_dtype)
        elif torch.is_complex(x):
            raise ValueError("Complex tensors are not supported")
        elif not torch.is_floating_point(x) and x.dtype != int_dtype:
            return x.to(int_dtype)

    return x


def convert_inputs(func, parameters=None):
    """Convert a function's inputs to the right type.

    It converts the inputs arrays to the right type (torch.Tensor) and
    convert the dtype of the tensor to the right one (float32 for float,
    int64 for int), before calling the function. It allows to use the
    function with numpy arrays or torch tensors, but keep in mind that
    scikit-shapes will always convert the inputs to torch tensors.

    If used in combination with the typecheck decorator, it must be called
    first :
    ```python
    import numpy as np
    import skshapes as sks
    from skshapes.types import NumericalTensor


    @sks.convert_inputs
    @sks.typecheck
    def foo(a: NumericalTensor) -> NumericalTensor:
        return a


    foo(np.zeros(10))  # OK


    @sks.typecheck
    @sks.convert_inputs
    def bar(a: NumericalTensor) -> NumericalTensor:
        return a


    bar(np.zeros(10))  # Beartype error
    ```

    TODO: so far, it only works with numpy arrays and torch tensors.
    Is it relevant to add support for lists and tuples ? -> must be careful
    on which arguments are converted (only the ones that are supposed to be
    converted to torch.Tensor).
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Convert args and kwargs to torch.Tensor
        # and convert the dtype to the right one
        new_args = []
        for arg in args:
            new_args.append(_convert_arg(arg))

        for key, value in kwargs.items():
            kwargs[key] = _convert_arg(value)

        return func(*new_args, **kwargs)

    return wrapper
