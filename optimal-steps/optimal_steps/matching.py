"""Matching: FPFH descriptors and target position of every source point (Section 2 of the paper)."""

from typing import Optional

import open3d.core as o3c
import open3d.t.geometry as tgeo
import open3d.t.pipelines.registration as treg
import torch
from torch.utils.dlpack import from_dlpack, to_dlpack


def compute_fpfh(
    points: torch.Tensor, normals: torch.Tensor, radius: float
) -> torch.Tensor:
    if not points.is_contiguous():
        points = points.contiguous()
    if not normals.is_contiguous():
        normals = normals.contiguous()

    device = points.device
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        o3d_device = o3c.Device(f"CUDA:{idx}")
    else:
        o3d_device = o3c.Device("CPU:0")

    pcd = tgeo.PointCloud(o3d_device)
    pcd.point["positions"] = o3c.Tensor.from_dlpack(to_dlpack(points.detach()))
    pcd.point["normals"] = o3c.Tensor.from_dlpack(to_dlpack(normals.detach()))

    fpfh_tensor = treg.compute_fpfh_feature(pcd, radius=radius, max_nn=100)
    out_tensor = from_dlpack(fpfh_tensor.to_dlpack())
    return torch.nn.functional.normalize(out_tensor, p=2, dim=1)


def effective_targets(
    points: torch.Tensor,
    forward_targets: torch.Tensor,
    target_points: torch.Tensor,
    backward_matches: Optional[torch.Tensor],
    trust_symmetric: float,
) -> torch.Tensor:
    """Target position z_i of every source point (Section 2 of the paper).

    points:           (N, 3) current source points x_i.
    forward_targets:  (N, 3) matched target points y_σ(i).
    target_points:    (M, 3) target points y_j.
    backward_matches: (M,) τ(j), index of the source point matched to each target
                      point, or None for forward matching only.
    trust_symmetric:  κ.

    z_i = x_i + (1 - κ) (y_σ(i) - x_i) + κ (t_i - x_i), where t_i is the barycentre of
    the target points whose nearest source point is x_i. For a source point that is
    the nearest neighbour of no target point, the last term is zero:
    z_i = x_i + (1 - κ) (y_σ(i) - x_i).
    """
    forward = forward_targets - points
    if backward_matches is None:
        return points + forward

    n_points = points.shape[0]
    backward = torch.zeros_like(points)
    backward.index_add_(0, backward_matches, target_points - points[backward_matches])
    counts = torch.zeros((n_points, 1), device=points.device)
    counts.index_add_(0, backward_matches, torch.ones((target_points.shape[0], 1), device=points.device))
    backward = backward / torch.clamp(counts, min=1.0)
    return points + ((1.0 - trust_symmetric) * forward + trust_symmetric * backward)
