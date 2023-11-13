"""Generic Multiscale class."""

from __future__ import annotations

from ..types import (
    typecheck,
    shape_type,
)
from typing import Union
from .multiscale_triangle_mesh import MultiscaleTriangleMesh
from ..decimation import Decimation


class Multiscale:
    """Generic multiscale class."""

    @typecheck
    def __new__(
        cls,
        shape: Union[shape_type, list[shape_type]],
        correspondence: bool = False,
        **kwargs,
    ) -> Union[MultiscaleTriangleMesh, list[MultiscaleTriangleMesh]]:
        """Multiscale object from a shape or a list of shapes.

        Depending on the type of the shape, the corresponding multiscale object
        is created (multiscale triangle mesh, multiscale point cloud, etc.)

        Parameters
        ----------
        shape
            A shape or a list of shapes.
        correspondence
            Wether the shapes of the list should be considered to be in
            pointwise correspondence or not.

        Returns
        -------
        Union[MultiscaleTriangleMesh,list[MultiscaleTriangleMesh]]
            A multiscale object or a list of multiscale objects.
        """
        if isinstance(shape, list) and not correspondence:
            # if no correspondence, do nothing more than call the constructor
            # independently on each shape
            return [cls(s, **kwargs) for s in shape]

        elif not isinstance(shape, list):
            # here Multiscale is called on a single shape, compute the
            # multiscale object depending on the type of the shape
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
            # here Multiscale is called on a list of shapes thath are supposed
            # to be corresponding to each other. The correspondence is used to
            # decimate the shapes in parallel.

            if (
                hasattr(shape[0], "is_triangle_mesh")
                and shape[0].is_triangle_mesh
            ):
                # Triangle meshes

                if "ratios" in kwargs.keys():
                    target_reduction = 1 - min(kwargs["ratios"])
                elif "n_points" in kwargs.keys():
                    target_reduction = 1 - min(kwargs["n_points"])

                # check that all shapes are triangle meshes and share the same
                # topology
                if not all(s.is_triangle_mesh for s in shape):
                    raise ValueError(
                        "All shapes must be triangle meshes to be decimated"
                        + " in correspondence"
                    )

                if not all(s.n_points == shape[0].n_points for s in shape):
                    raise ValueError(
                        "All shapes must have the same number of points to be"
                        + " decimated in correspondence"
                    )

                if not all(
                    s.n_triangles == shape[0].n_triangles for s in shape
                ):
                    raise ValueError(
                        "All shapes must have the same number of triangles to"
                        + " be decimated in correspondence"
                    )

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
