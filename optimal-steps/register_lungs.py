"""Lung vascular tree registration on Lung250M-4B (paper, Section 3).

Registers the two vessel point clouds of each Lung250M-4B case and measures the
target registration error (TRE) on the landmarks of the case (about 100 per case).

Data: Lung250M-4B (Falta et al., NeurIPS 2023), https://github.com/multimodallearning/Lung250M-4B:
the point clouds of the test cases ("cloudsTs", from the download link of that page) and
the landmarks ("evaluation/lms_validation.pth" in that repository).

Expected data layout (Lung250M-4B point clouds and validation landmarks):

    <data_dir>/coordinates/case_XXX_1.pth   source cloud (fixed / moving pair: _1 -> _2)
    <data_dir>/coordinates/case_XXX_2.pth   target cloud
    <data_dir>/artery_vein/case_XXX_{1,2}.pth   artery / vein label per point
    <data_dir>/distance/case_XXX_{1,2}.pth      vessel radius (distance transform) per point
    <landmarks>                                 dict {case number: (100, 6) tensor [source xyz | target xyz]}

Each .pth file holds a list of three clouds: [8k points, skeletonized (25k-60k points), full];
the skeletonized cloud (index 1) is the one used in the paper.

    python register_lungs.py --data_dir cloudsTs --landmarks lms_validation.pth
    python register_lungs.py --data_dir cloudsTs --landmarks lms_validation.pth --case_id case_056

The deformation model is the one of optimal_steps (same exponential kernel). Vessel
trees are point clouds without normals, so this experiment uses:
- a matching score that combines position, vessel radius, vessel direction and
  artery/vein label (VesselTreeRegistration.compute_correspondences), whose radius and
  direction terms are relaxed from the first to the last scale,
- an un-normalised regularisation weight (no rescaling by the number of points).
"""

import argparse
import csv
import glob
import os
import time

# optimal_steps must be imported before pykeops: it sets up the compiler KeOps uses on macOS.
from optimal_steps import DiffeomorphicRegistration, RegistrationConfig, resolve_device

# isort: split

import numpy as np
import open3d as o3d
import pyvista as pv
import torch
from pykeops.torch import LazyTensor
from scipy.spatial import cKDTree

RESOLUTION_INDEX = 1  # skeletonized cloud, 25k-60k points


LUNG_CONFIG = dict(
    sigma_init=50.0,
    sigma_final=1.25,
    n_scales=5,
    outer_steps=8,
    lambda_reg=1.0,
    euler_precision_step_mm=0.1,
    metric_type="point2point",
    use_fpfh=False,
    normal_weight=0.0,
    use_sinkhorn=True,
    use_symmetric_correspondences=True,
    trust_symmetric=0.5,
)


class VesselTreeRegistration(DiffeomorphicRegistration):
    """DiffeomorphicRegistration with a vessel-aware matching score for point clouds.

    Per-point attributes: artery/vein label `av` (N, 1), vessel radius `radius`
    (N, 1) and unit vessel direction `tangent` (N, 3), for source and target.
    """

    SIGMA_AV = 0.1  # tolerance on the artery/vein label

    def __init__(self, source_cloud, target_cloud, config, source_attrs, target_attrs):
        super().__init__(source_cloud, target_cloud, config)
        # Regularisation weight not rescaled by the number of points, as in the
        # experiments of the paper.
        self.lambda_regs = [float(config.lambda_reg)] * self.n_scales

        def as_tensor(x):
            return torch.as_tensor(x, dtype=torch.float32, device=self.device).contiguous()

        self.src_av, self.src_radius, self.src_tangent = map(as_tensor, source_attrs)
        self.tgt_av, self.tgt_radius, self.tgt_tangent = map(as_tensor, target_attrs)

    def _relaxation(self) -> float:
        # 1 at the coarsest scale, 0 at the finest one.
        scale_idx = int(np.argmin(np.abs(self.sigmas - self.current_sigma)))
        return 1.0 - scale_idx / max(1, self.n_scales - 1)

    def compute_correspondences(self, q, current_src_normals=None, precomputed_src_fpfh=None):
        relax = self._relaxation()

        def scalar(v):
            return LazyTensor(torch.tensor([v], dtype=torch.float32, device=self.device))

        d_ij = ((LazyTensor(q[:, None, :]) - LazyTensor(self.tgt[None, :, :])) ** 2).sum(-1)
        r_i = LazyTensor(self.src_radius[:, None, :])
        r_j = LazyTensor(self.tgt_radius[None, :, :])
        t_i = LazyTensor(self.src_tangent[:, None, :])
        t_j = LazyTensor(self.tgt_tangent[None, :, :])
        av_i = LazyTensor(self.src_av[:, None, :])
        av_j = LazyTensor(self.tgt_av[None, :, :])

        # Spatial tolerance widened by the vessel radii at coarse scales.
        sigma_sq = scalar(self.current_sigma) ** 2 + (r_i**2 + r_j**2) * relax
        # Vessel directions are unoriented: compare them through <t_i, t_j>^2.
        alignment = (1.0 - relax) + relax * (t_i * t_j).sum(-1) ** 2
        radius_penalty = (r_i - r_j) ** 2 / scalar(1.0 + 9.0 * (1.0 - relax)) ** 2
        av_penalty = (av_i - av_j) ** 2 / scalar(self.SIGMA_AV) ** 2

        similarity = (-d_ij / sigma_sq - radius_penalty - av_penalty).exp() * alignment

        idx_s_to_t = similarity.argmax(dim=1).long().view(-1)
        idx_t_to_s = None
        if self.use_symmetric_correspondences:
            idx_t_to_s = similarity.argmax(dim=0).long().view(-1)
        return self.tgt[idx_s_to_t], self.tgt_normals[idx_s_to_t], idx_t_to_s


def pre_align(source: torch.Tensor, target: torch.Tensor, voxel_size: float = 10.0) -> np.ndarray:
    """Centroid matching, anisotropic scaling of the robust extents, then ICP."""
    src = source.numpy()
    tgt = target.numpy()

    tgt_centroid = tgt.mean(axis=0)
    centered = src - src.mean(axis=0) + tgt_centroid
    src_extent = np.percentile(centered, 99, axis=0) - np.percentile(centered, 1, axis=0)
    tgt_extent = np.percentile(tgt, 99, axis=0) - np.percentile(tgt, 1, axis=0)
    scaled = tgt_centroid + (centered - tgt_centroid) * (tgt_extent / (src_extent + 1e-8))

    src_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scaled))
    tgt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt))
    icp = o3d.pipelines.registration.registration_icp(
        src_pcd.voxel_down_sample(voxel_size),
        tgt_pcd.voxel_down_sample(voxel_size),
        voxel_size * 4,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )
    src_pcd.transform(icp.transformation)
    return np.asarray(src_pcd.points, dtype=np.float32)


def vessel_tangents(points: np.ndarray, radius: float = 5.0) -> np.ndarray:
    """Main local PCA direction in a ball of `radius` mm around each point."""
    tree = cKDTree(points)
    tangents = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (len(points), 1))
    for i, neighbours in enumerate(tree.query_ball_point(points, r=radius)):
        if len(neighbours) < 3:
            continue
        local = points[neighbours] - points[neighbours].mean(axis=0)
        _, eigvecs = np.linalg.eigh(local.T @ local)
        tangents[i] = eigvecs[:, -1]
    return tangents


def _load(path: str) -> torch.Tensor:
    return torch.load(path, weights_only=True)[RESOLUTION_INDEX].float()


def _match_length(values: torch.Tensor, n: int) -> np.ndarray:
    """Per-point attribute as an (n, 1) array, truncated or zero-padded if needed."""
    v = values.reshape(-1).float().numpy()
    if len(v) != n:
        print(f"  [warning] {len(v)} attribute values for {n} points, truncating/padding")
        v = np.pad(v[:n], (0, max(0, n - len(v))))
    return v.reshape(-1, 1)


def load_case(data_dir: str, case: str, suffix: str):
    points = _load(os.path.join(data_dir, "coordinates", f"{case}_{suffix}.pth"))
    n = len(points)
    av = _match_length(_load(os.path.join(data_dir, "artery_vein", f"{case}_{suffix}.pth")), n)
    radius = _match_length(_load(os.path.join(data_dir, "distance", f"{case}_{suffix}.pth")), n)
    return points, av, radius


def register_clouds(src, tgt, src_av, src_r, tgt_av, tgt_r, config: RegistrationConfig, return_history=False):
    """Register two vessel point clouds (torch tensors (N, 3), attributes (N, 1)).

    Returns the displacement of every source point; it is defined on the original
    source points, so it includes the pre-alignment. With return_history=True, also
    returns the source points after the pre-alignment and after every Gauss-Newton
    step, as an array of shape (1 + n_scales * outer_steps, N, 3).
    """
    src_aligned = pre_align(src, tgt)
    # Directions are estimated on the raw clouds; the ICP rotation is small
    # and <t_i, t_j>^2 is insensitive to the orientation of the tangents.
    src_t = vessel_tangents(src.numpy())
    tgt_t = vessel_tangents(tgt.numpy())

    model = VesselTreeRegistration(
        pv.PolyData(src_aligned),
        pv.PolyData(tgt.numpy()),
        config,
        source_attrs=(src_av, src_r, src_t),
        target_attrs=(tgt_av, tgt_r, tgt_t),
    )
    if not return_history:
        return model.run().cpu() - src
    deformed, trajectory, _ = model.run(return_history=True)
    return deformed.cpu() - src, trajectory.cpu().numpy()


def register_case(data_dir: str, case: str, config: RegistrationConfig):
    """Load and register one Lung250M-4B case: (source, target, displacement)."""
    src, src_av, src_r = load_case(data_dir, case, "1")
    tgt, tgt_av, tgt_r = load_case(data_dir, case, "2")
    return src, tgt, register_clouds(src, tgt, src_av, src_r, tgt_av, tgt_r, config)


def interpolate_displacements(points, displacements, queries, k=15):
    """Gaussian-weighted average of the displacements of the k nearest points,
    with a bandwidth set to the mean distance to those neighbours (>= 1 mm)."""
    dist, idx = cKDTree(points).query(queries, k=k)
    sigma = np.maximum(dist.mean(axis=1, keepdims=True), 1.0)
    w = np.maximum(np.exp(-(dist**2) / (2.0 * sigma**2)), 1e-12)
    w /= w.sum(axis=1, keepdims=True)
    return np.sum(displacements[idx] * w[..., None], axis=1)


def landmark_errors(source_points, displacement, landmarks):
    """TRE (mm) before and after registration for a (L, 6) landmark array."""
    lm_src, lm_tgt = landmarks[:, :3], landmarks[:, 3:]
    warped = lm_src + interpolate_displacements(source_points, displacement, lm_src)
    return np.linalg.norm(lm_tgt - lm_src, axis=1), np.linalg.norm(lm_tgt - warped, axis=1)


def save_clouds(out_dir, case, src, tgt, displacement):
    for name, pts in (("source", src), ("target", tgt), ("deformed", src + displacement)):
        pv.PolyData(np.asarray(pts, dtype=np.float32)).save(os.path.join(out_dir, f"{case}_{name}.vtp"))


def main():
    parser = argparse.ArgumentParser(
        description="Lung250M-4B vessel tree registration + landmark TRE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", default="cloudsTs", help="Folder with coordinates/, artery_vein/, distance/")
    parser.add_argument("--landmarks", default="lms_validation.pth", help="Landmark dict (.pth)")
    parser.add_argument("--output_dir", default="results/lungs")
    parser.add_argument("--case_id", nargs="+", default=None, help="e.g. case_056 (default: all cases)")
    parser.add_argument("--save_vtp", action="store_true", help="Also save source/target/deformed clouds as .vtp")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    config = RegistrationConfig(**LUNG_CONFIG, device=args.device)
    resolve_device(config.device)

    if args.case_id:
        cases = args.case_id
    else:
        pattern = os.path.join(args.data_dir, "coordinates", "*_1.pth")
        cases = sorted(os.path.basename(p)[: -len("_1.pth")] for p in glob.glob(pattern))
    if not cases:
        raise SystemExit(f"No case found in {args.data_dir}/coordinates")

    landmarks = torch.load(args.landmarks, weights_only=True) if os.path.exists(args.landmarks) else None
    if landmarks is None:
        print(f"[warning] {args.landmarks} not found: registering without TRE evaluation")

    pred_dir = os.path.join(args.output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    rows, all_init, all_final = [], [], []
    for i, case in enumerate(cases):
        print(f"\n[{i + 1}/{len(cases)}] {case}", flush=True)
        t0 = time.perf_counter()
        src, tgt, disp = register_case(args.data_dir, case, config)
        elapsed = time.perf_counter() - t0
        torch.save(disp, os.path.join(pred_dir, f"{case}.pth"))
        if args.save_vtp:
            save_clouds(pred_dir, case, src.numpy(), tgt.numpy(), disp.numpy())

        row = {"case": case, "n_source": len(src), "n_target": len(tgt), "time_s": round(elapsed, 2)}
        key = str(int(case.split("_")[-1]))
        if landmarks is not None and key in landmarks:
            init, final = landmark_errors(src.numpy(), disp.numpy(), landmarks[key].numpy())
            all_init.append(init)
            all_final.append(final)
            row.update(tre_initial_mm=init.mean(), tre_final_mm=final.mean(), tre_final_std_mm=final.std())
            print(f"  TRE {init.mean():.2f} mm -> {final.mean():.2f} ± {final.std():.2f} mm ({elapsed:.1f} s)", flush=True)
        rows.append(row)

        with open(os.path.join(args.output_dir, "tre.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(max(rows, key=len).keys()))
            writer.writeheader()
            writer.writerows(rows)

    if all_final:
        init, final = np.concatenate(all_init), np.concatenate(all_final)
        print(
            f"\n{len(all_final)} cases: TRE {init.mean():.2f} mm before, "
            f"{final.mean():.2f} ± {final.std():.2f} mm after registration"
        )
    print(f"Results in {args.output_dir}/ (tre.csv, predictions/)")


if __name__ == "__main__":
    main()
