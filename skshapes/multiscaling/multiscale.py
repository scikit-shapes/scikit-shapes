"""This module contains the multiscale class."""
# TODO : add_points_data interface ? Deal with existing data ?
# TODO : landmark interface ?


# Signal management :
# maintain a dict of signals/policy
# when the multiscale is initialized, the list corresponds to the signals at the origin ratio
# when a ratio is added, the signals are propagated to the new ratio
# when at is called, the signal is propagated to the given ratio

from __future__ import annotations

from ..types import (
    typecheck,
    convert_inputs,
    Optional,
    FloatSequence,
    IntSequence,
    int_dtype,
    float_dtype,
    Int1dTensor,
    NumericalTensor,
    Number,
    polydata_type,
    shape_type,
)
from typing import Literal
from ..utils import scatter
import torch
from typing import Union
from .multiscale_triangle_mesh import MultiscaleTriangleMesh
from ..decimation import Decimation


class Multiscale:
    @typecheck
    def __new__(
        cls,
        shape: Union[shape_type, list[shape_type]],
        correspondence: bool = False,
        **kwargs,
    ) -> Union[MultiscaleTriangleMesh, list[MultiscaleTriangleMesh]]:
        """Create a multiscale object from a shape or a list of shapes.

        Args:
            shape (Union[shape_type,list[shape_type]]): A shape or a list of shapes.
            correspondence (bool, optional): Wether the shapes of the list should be considered to be in correspondence or not.

        Returns:
            Union[MultiscaleTriangleMesh,list[MultiscaleTriangleMesh]]
        """
        if isinstance(shape, list) and not correspondence:
            # if no correspondence, do nothing more than call the constructor
            # independently on each shape
            return [cls(s, **kwargs) for s in shape]

        elif not isinstance(shape, list):
            # here Multiscale is called on a single shape, compute the multiscale object
            # depending on the type of the shape
            if hasattr(shape, "is_triangle_mesh") and shape.is_triangle_mesh:
                instance = super(Multiscale, cls).__new__(
                    MultiscaleTriangleMesh
                )
                instance.__init__(shape=shape, **kwargs)
                return instance

            else:
                raise NotImplementedError(
                    "Only triangle meshes are supported for now"
                )

        elif isinstance(shape, list) and correspondence:
            # here Multiscale is called on a list of shapes thath are supposed to be
            # corresponding to each other. The correspondence is used to decimate
            # the shapes in parallel.

            if (
                hasattr(shape[0], "is_triangle_mesh")
                and shape[0].is_triangle_mesh
            ):
                # Triangle meshes

                if "ratios" in kwargs.keys():
                    target_reduction = 1 - min(kwargs["ratios"])
                elif "n_points" in kwargs.keys():
                    target_reduction = 1 - min(kwargs["n_points"])

                # check that all shapes are triangle meshes and share the same topology
                assert all(
                    [s.is_triangle_mesh for s in shape]
                ), "All shapes must be triangle meshes to be used with correspondence"
                assert all(
                    [s.n_points == shape[0].n_points for s in shape]
                ), "All shapes must have the same number of points to be decimated in correspondence"
                assert all(
                    [
                        torch.allclose(s.triangles, shape[0].triangles)
                        for s in shape
                    ]
                ), "All shapes must have the same triangles to be decimated in correspondence"

                # compute the decimation module
                decimation_module = Decimation(
                    target_reduction=target_reduction
                )
                decimation_module.fit(shape[0])

                # add the decimation module to the kwargs
                kwargs["decimation_module"] = decimation_module

                return [cls(s, **kwargs) for s in shape]

            else:
                raise NotImplementedError(
                    "Only triangle meshes are supported for now"
                )

        else:
            if hasattr(shape, "is_triangle_mesh") and shape.is_triangle_mesh:
                instance = super(Multiscale, cls).__new__(
                    MultiscaleTriangleMesh
                )
                instance.__init__(shape=shape, **kwargs)
                return instance

            else:
                raise NotImplementedError(
                    "Only triangle meshes are supported for now"
                )
