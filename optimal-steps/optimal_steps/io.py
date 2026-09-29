"""Loading of meshes, point clouds and segmentations (surface extraction)."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pyvista as pv
import torch
import torch.nn.functional as F

from .config import resolve_device


def gaussian_smooth_gpu(tensor, sigma_mm, spacing, truncate=2.0):
    if not isinstance(tensor, torch.Tensor):
        device = resolve_device("auto")
        tensor = torch.tensor(tensor, dtype=torch.float32, device=device)
    else:
        tensor = tensor.float()

    sigmas = np.array(sigma_mm) / np.array(spacing)
    current = tensor.unsqueeze(0).unsqueeze(0)

    for i, s in enumerate(sigmas):
        if s < 1e-3:
            continue
        radius = int(truncate * s + 0.5)
        kernel_size = 2 * radius + 1
        x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=tensor.device)
        k = torch.exp(-(x**2) / (2 * s**2))
        k = k / k.sum()
        shape = [1, 1, 1, 1, 1]
        shape[2 + i] = kernel_size
        k = k.view(*shape)
        try:
            current = F.conv3d(current, k, padding="same")
        except NotImplementedError:
            pad_amount = radius
            paddings = [0] * 6
            pad_idx = 2 * (2 - i)
            paddings[pad_idx] = pad_amount
            paddings[pad_idx + 1] = pad_amount
            current = F.pad(current, paddings, mode="replicate")
            current = F.conv3d(current, k)

    return current.squeeze().cpu().numpy()


def _extract_mesh_from_volume(
    vol: pv.ImageData,
    label: int,
    max_points: Optional[int] = None,
    smooth_sigma_mm: float = 0.5,
    padding: int = 10,
) -> pv.PolyData:
    vol_array = vol.active_scalars.reshape(vol.dimensions, order="F")

    step = 3
    vol_small = vol_array[::step, ::step, ::step]
    coords_small = np.argwhere(vol_small == label)
    if coords_small.size == 0:
        raise ValueError(f"Label {label} not found in volume!")

    min_idx = coords_small.min(axis=0) * step
    max_idx = (coords_small.max(axis=0) + 1) * step
    min_idx = np.maximum(min_idx - padding, 0)
    max_idx = np.minimum(max_idx + padding, np.array(vol.dimensions) - 1)

    slices = tuple(slice(min_idx[i], max_idx[i]) for i in range(3))
    cropped_vol = vol_array[slices]

    arr_smooth = (cropped_vol == label).astype(np.float32)
    current_spacing = np.array(vol.spacing)
    if smooth_sigma_mm is None:
        smooth_sigma_mm = 1.5 * np.min(current_spacing)

    arr_smooth = gaussian_smooth_gpu(
        arr_smooth, smooth_sigma_mm, current_spacing, truncate=2.0
    )

    min_idx_final = np.array([s.start for s in slices])
    crop_origin = np.array(vol.origin) + (min_idx_final * current_spacing)

    grid = pv.ImageData()
    grid.dimensions = arr_smooth.shape
    grid.spacing = current_spacing
    grid.origin = crop_origin
    grid.point_data["scalars"] = arr_smooth.flatten(order="F")

    mesh = grid.contour([0.5], scalars="scalars")

    if max_points is not None and mesh.n_points > max_points:
        mesh = mesh.decimate(1 - (max_points / mesh.n_points))

    return mesh


_POINTCLOUD_EXTENSIONS = {".xyz", ".pts", ".csv"}
_VOLUME_EXTENSIONS = {".nii", ".nii.gz", ".mha", ".mhd"}


def load_input(
    path: str,
    label: Optional[int] = None,
    max_points: Optional[int] = None,
) -> Tuple[pv.PolyData, bool]:

    p = Path(path)
    suffix = "".join(p.suffixes).lower()
    if not suffix:
        suffix = p.suffix.lower()

    if suffix in _VOLUME_EXTENSIONS:
        if label is None:
            raise ValueError(
                f"Volume input '{path}' requires --source_label / --target_label."
            )
        vol = pv.read(path)
        mesh = _extract_mesh_from_volume(vol, label, max_points=max_points)
        if not mesh.is_all_triangles:
            mesh = mesh.triangulate()
        return mesh, True

    if p.suffix.lower() in _POINTCLOUD_EXTENSIONS:
        data = np.loadtxt(path, delimiter="," if p.suffix.lower() == ".csv" else None)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        points = data[:, :3].astype(np.float32)
        if max_points is not None and len(points) > max_points:
            idx = np.random.choice(len(points), max_points, replace=False)
            points = points[idx]
        mesh = pv.PolyData(points)
        return mesh, False

    mesh = pv.read(path)
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    mesh = mesh.clean()
    if mesh.n_faces_strict == 0:
        # File loaded as a point cloud (no face connectivity)
        points = np.asarray(mesh.points, dtype=np.float32)
        if max_points is not None and len(points) > max_points:
            idx = np.random.choice(len(points), max_points, replace=False)
            points = points[idx]
        return pv.PolyData(points), False
    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()
    if max_points is not None and mesh.n_points > max_points:
        mesh = mesh.decimate(1 - (max_points / mesh.n_points))
    return mesh, True
