"""Rigid + anisotropic pre-alignment (RANSAC on FPFH features, then ICP)."""

import logging
from typing import Tuple

import numpy as np
import open3d as o3d
import pyvista as pv


def align_rigid(
    source: pv.PolyData,
    target: pv.PolyData,
    voxel_size: float = 2.0,
) -> Tuple[pv.PolyData, np.ndarray, np.ndarray]:

    logger = logging.getLogger(__name__)

    if o3d.core.cuda.device_count() > 0:
        device = o3d.core.Device("CUDA:0")
    else:
        device = o3d.core.Device("CPU:0")

    source_t = o3d.t.geometry.PointCloud(device)
    source_t.point.positions = o3d.core.Tensor(
        np.asarray(source.points), o3d.core.float32, device
    )
    target_t = o3d.t.geometry.PointCloud(device)
    target_t.point.positions = o3d.core.Tensor(
        np.asarray(target.points), o3d.core.float32, device
    )

    source_down = source_t.voxel_down_sample(voxel_size)
    target_down = target_t.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    source_down.estimate_normals(radius=radius_normal, max_nn=30)
    target_down.estimate_normals(radius=radius_normal, max_nn=30)

    radius_feature = voxel_size * 5
    source_fpfh_t = o3d.t.pipelines.registration.compute_fpfh_feature(
        source_down, radius=radius_feature, max_nn=100
    )
    target_fpfh_t = o3d.t.pipelines.registration.compute_fpfh_feature(
        target_down, radius=radius_feature, max_nn=100
    )

    source_down_legacy = source_down.to_legacy()
    target_down_legacy = target_down.to_legacy()

    source_fpfh_legacy = o3d.pipelines.registration.Feature()
    source_fpfh_legacy.data = source_fpfh_t.cpu().numpy().T
    target_fpfh_legacy = o3d.pipelines.registration.Feature()
    target_fpfh_legacy.data = target_fpfh_t.cpu().numpy().T

    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down_legacy,
        target_down_legacy,
        source_fpfh_legacy,
        target_fpfh_legacy,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.6),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(10000, 0.99),
    )

    source_t.estimate_normals(radius=radius_normal, max_nn=30)
    target_t.estimate_normals(radius=radius_normal, max_nn=30)
    init_trans = o3d.core.Tensor(result.transformation, o3d.core.float64, device)
    result_icp = o3d.t.pipelines.registration.icp(
        source_t,
        target_t,
        distance_threshold,
        init_trans,
        o3d.t.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    rigid_transform = result_icp.transformation.cpu().numpy()

    aligned_source = source.copy()
    aligned_source.transform(rigid_transform, inplace=True)

    src_pts = np.asarray(aligned_source.points)
    tgt_pts = np.asarray(target.points)
    src_extent = np.max(src_pts, axis=0) - np.min(src_pts, axis=0)
    tgt_extent = np.max(tgt_pts, axis=0) - np.min(tgt_pts, axis=0)
    scale_factors = tgt_extent / (src_extent + 1e-8)
    logger.info(f"Anisotropic scale factors (x, y, z): {scale_factors}")

    src_centroid = np.mean(src_pts, axis=0)
    scaled_pts = src_centroid + (src_pts - src_centroid) * scale_factors
    aligned_source.points = scaled_pts

    return aligned_source, rigid_transform, scale_factors
