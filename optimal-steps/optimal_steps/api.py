"""register(): pre-alignment and diffeomorphic registration of a source onto a target."""

import time
from dataclasses import dataclass, fields
from typing import Optional, Union

import numpy as np
import pyvista as pv
import torch

from .alignment import align_rigid
from .config import RegistrationConfig
from .geometry import add_xy_tcoords_vtk
from .io import load_input
from .model import DiffeomorphicRegistration


@dataclass
class RegistrationResult:
    deformed_points: np.ndarray
    source_mesh: pv.PolyData
    deformed_mesh: pv.PolyData
    rigid_transform: Optional[np.ndarray]
    scale_factors: Optional[np.ndarray]
    timings: dict
    trajectory_q: Optional[np.ndarray] = None
    model: Optional[DiffeomorphicRegistration] = None  # with return_history=True
    target_mesh: Optional[pv.PolyData] = None


def register(
    source: Union[str, pv.PolyData],
    target: Union[str, pv.PolyData],
    source_landmark_indices: Optional[np.ndarray] = None,
    target_landmark_indices: Optional[np.ndarray] = None,
    config: Optional[RegistrationConfig] = None,
    source_label: Optional[int] = None,
    target_label: Optional[int] = None,
    max_points: Optional[int] = None,
    rigid_align: bool = True,
    return_history: bool = False,
    source_features: Optional[Union[np.ndarray, torch.Tensor]] = None,
    target_features: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> RegistrationResult:

    if config is None:
        config = RegistrationConfig()

    timings: dict = {}

    t0 = time.perf_counter()
    if isinstance(source, str):
        src_mesh, src_has_conn = load_input(
            source,
            label=source_label,
            max_points=None,  # we don't want to decimate the template
        )
    else:
        src_mesh = source
        src_has_conn = True

    if isinstance(target, str):
        tgt_mesh, tgt_has_conn = load_input(
            target, label=target_label, max_points=max_points
        )
    else:
        tgt_mesh = target
        tgt_has_conn = True

    timings["data_loading"] = time.perf_counter() - t0

    if src_has_conn:
        add_xy_tcoords_vtk(src_mesh)

    has_connectivity = src_has_conn and tgt_has_conn

    if not has_connectivity:
        import warnings

        warnings.warn(
            "Input is a raw point cloud: FPFH, normals and plane metrics disabled.",
            UserWarning,
        )
        config = RegistrationConfig(
            **{
                **{f.name: getattr(config, f.name) for f in fields(config)},
                "use_fpfh": False,
                "metric_type": "point2point",
                "normal_weight": 0.0,
            }
        )

    rigid_transform = None
    scale_factors = None
    if rigid_align:
        t0 = time.perf_counter()
        src_mesh, rigid_transform, scale_factors = align_rigid(src_mesh, tgt_mesh)
        timings["rigid_alignment"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    reg = DiffeomorphicRegistration(
        src_mesh,
        tgt_mesh,
        config,
        source_landmark_indices=source_landmark_indices,
        target_landmark_indices=target_landmark_indices,
        source_features=source_features,
        target_features=target_features,
    )
    result = reg.run(return_history=return_history)
    timings["non_rigid"] = time.perf_counter() - t0

    trajectory_q = None
    model = None
    if return_history:
        warped_pts_tensor, traj_q, traj_p = result
        trajectory_q = traj_q.detach().cpu().numpy()
        model = reg
    else:
        warped_pts_tensor = result

    warped_pts = warped_pts_tensor.detach().cpu().numpy()
    deformed_mesh = src_mesh.copy()
    deformed_mesh.points = warped_pts

    return RegistrationResult(
        deformed_points=warped_pts,
        source_mesh=src_mesh,
        deformed_mesh=deformed_mesh,
        rigid_transform=rigid_transform,
        scale_factors=scale_factors,
        timings=timings,
        trajectory_q=trajectory_q,
        model=model,
        target_mesh=tgt_mesh,
    )
