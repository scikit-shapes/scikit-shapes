"""Converters for arguments."""

import itertools
from functools import wraps
from inspect import isclass, signature
from types import UnionType
from typing import Union, get_args, get_origin, get_type_hints

import jaxtyping
import numpy as np
import torch


def detect_array_dtypes(t):
    if get_origin(t) in [Union, UnionType]:
        # If type is a Union, we iterate through the types
        # and return a list of acceptable dtypes, without duplicates
        return list(
            set(
                itertools.chain(*[detect_array_dtypes(a) for a in get_args(t)])
            )
        )

    # We only bother converting to specific dtypes.
    # Notably, landmarks may be specified with "Int" and are not affected.
    elif isclass(t) and issubclass(t, jaxtyping.AbstractArray):
        if len(t.dtypes) == 1 and t.dtypes[0] in [
            "float32",
            "float64",
            "int64",
        ]:
            return list(t.dtypes)
        else:
            return []

    else:
        return []


def closest_dtype(dtype, target_dtypes):
    if target_dtypes == ["float32"]:
        return torch.float32
    elif target_dtypes == ["int64"]:
        return torch.int64
    elif sorted(target_dtypes) == ["float32", "int64"]:
        if dtype in [torch.float32, torch.float64]:
            return torch.float32
        elif dtype in [torch.int32, torch.int64]:
            return torch.int64
        else:
            return dtype
    else:
        msg = f"Unsupported target dtype: {target_dtypes}"
        raise NotImplementedError(msg)


def convert_inputs(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        sig = signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Iterate through the function's parameters and type hints
        for param_name, param_type in get_type_hints(func).items():

            # Do not waste time on default arguments
            if param_name in bound_args.arguments:

                # Detect if type hints require a specific float32 and/or int64 dtype
                # More vague types (e.g. "Int") are not converted
                target_dtypes = detect_array_dtypes(param_type)
                if target_dtypes:  # is not []

                    # At this point, we know that the parameter has been set
                    # and that it is supposed to be a torch.Tensor.
                    # If it is a NumPy array, a torch Tensor (maybe with incorrect dtype),
                    # a list or a tuple, we convert it to a torch.Tensor with the correct
                    # dtype. Otherwise, we may be in a situation where the param_type
                    # is a Union of different types (e.g. points in PolyData, that
                    # can also be the path to a file). In this case, we do not attempt
                    # to convert anything, and let the function or beartype raise an error
                    # if the input is not of the right type.
                    value = bound_args.arguments[param_name]

                    # We attempt to convert lists, tuples, numpy arrays and torch tensors
                    # that do not have the correct dtype.
                    if isinstance(value, list | tuple):
                        value = torch.tensor(value)

                    if isinstance(value, np.ndarray):
                        value = torch.from_numpy(value)

                    if isinstance(value, torch.Tensor):
                        if torch.is_complex(value):
                            msg = "Complex tensors are not supported"
                            raise ValueError(msg)

                        dtype = closest_dtype(value.dtype, target_dtypes)
                        bound_args.arguments[param_name] = value.to(
                            dtype=dtype
                        )
                    # Note that other types of "value" (e.g. a string) are not converted

        return func(*bound_args.args, **bound_args.kwargs)

    return wrapper
