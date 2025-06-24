from typing import Any

import numpy as np
import torch
from sklearn.neighbors import KDTree
from torch_cluster import knn_graph

torch.set_default_dtype(torch.float32)

try:
    import pyvista as pv

    HAS_PYVISTA = True
except ImportError as err:
    HAS_PYVISTA = False
    PYVISTA_WARNING = "pyvista is not installed. Please install it."
    raise ImportError(PYVISTA_WARNING) from err
try:
    import open3d as o3d

    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    OPEN3D_WARNING = "open3d is not installed. Please install it."


def _extract_points_and_normals(
    shape: pv.PolyData | torch.Tensor | np.ndarray | tuple,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Extract points and normals from various input formats."""
    if HAS_PYVISTA and isinstance(
        shape, pv.PolyData
    ):  # if the shape is a PyVista PolyData
        points = torch.from_numpy(shape.points).float()
        normals = None
        if "Normals" in shape.point_data:
            normals = torch.from_numpy(shape.point_data["Normals"]).float()
        elif (
            hasattr(shape, "point_normals") and shape.point_normals is not None
        ):
            normals = torch.from_numpy(shape.point_normals).float()
        return points, normals
    elif isinstance(
        shape, torch.Tensor | np.ndarray
    ):  # if the shape is a PyTorch or NumPy tensor (e.g., point cloud)
        if isinstance(shape, np.ndarray):
            shape = torch.from_numpy(shape).float()
        return (
            shape,
            None,
        )  # implement normal computation from Blanche Buet article
    elif (
        isinstance(shape, tuple) and len(shape) == 2
    ):  # if the shape is a tuple of (points, normals)
        points, normals = shape
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points).float()
        if isinstance(normals, np.ndarray):
            normals = torch.from_numpy(normals).float()
        return points, normals
    else:
        UNSUPPORTED_FORMAT_WARNING = (
            "Unsupported shape format. Expected PyVista PolyData, "
            "PyTorch/NumPy tensor, or tuple of (points, normals)."
        )
        raise ValueError(UNSUPPORTED_FORMAT_WARNING)


def _compute_local_metric(
    target_points: torch.Tensor,
    target_normals: torch.Tensor | None,
    metric_type: str = "point_to_plane",
    alpha: float = 1.0,
    beta: float = 0.5,
    source_points: torch.Tensor | None = None,  # noqa: ARG001
    source_normals: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute local metric matrices L_i."""
    n_points = target_points.shape[0]
    device = target_points.device

    if metric_type == "point_to_plane" and target_normals is not None:
        In33 = (
            torch.eye(3, device=device).unsqueeze(0).repeat(n_points, 1, 1)
        )  # (n_points, 3, 3)
        target_normals = target_normals / torch.norm(
            target_normals, dim=1, keepdim=True
        )
        nn_T = torch.bmm(
            target_normals.unsqueeze(2), target_normals.unsqueeze(1)
        )
        return alpha * nn_T + beta * (In33 - nn_T)

    elif (
        metric_type == "plane_to_plane"
        and target_normals is not None
        and source_normals is not None
    ):
        source_normals_norm = source_normals / torch.norm(
            source_normals, dim=1, keepdim=True
        )
        target_normals_norm = target_normals / torch.norm(
            target_normals, dim=1, keepdim=True
        )

        source_nn_T = torch.bmm(
            source_normals_norm.unsqueeze(2), source_normals_norm.unsqueeze(1)
        )

        target_nn_T = torch.bmm(
            target_normals_norm.unsqueeze(2), target_normals_norm.unsqueeze(1)
        )

        L_combined = alpha * (source_nn_T + target_nn_T)

        return L_combined  # noqa: RET504

    else:
        # Fallback to point-to-point
        return torch.eye(3, device=device).unsqueeze(0).repeat(n_points, 1, 1)


def _robust_loss_weights(
    residuals: torch.Tensor, loss_type: str = "cauchy_mad", **kwargs
) -> torch.Tensor:
    """Compute robust weights from residuals."""
    eps = kwargs.get("eps", 1e-8)

    if loss_type == "l1":
        return 1.0 / (torch.abs(residuals) + eps)

    elif loss_type == "var_trim":
        trim_ratio = kwargs.get("trim_ratio", 0.8)
        threshold = torch.quantile(residuals, trim_ratio)
        return (residuals <= threshold).to(dtype=residuals.dtype)

    elif loss_type == "cauchy":
        k = kwargs.get("k", 1.0)
        return 1.0 / (1.0 + (residuals / k) ** 2)

    elif loss_type == "cauchy_mad":
        median_res = torch.median(residuals)
        mad = torch.median(torch.abs(residuals - median_res))
        s = 1.4826 * mad
        k = kwargs.get("k", 1.0)
        normalized_res = residuals / (s + eps)
        return 1.0 / (1.0 + (normalized_res / k) ** 2)

    else:
        return torch.ones_like(residuals)


def _gauss_newton_gicp(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    correspondences: torch.Tensor,
    L_target: torch.Tensor,
    R_init: torch.Tensor,
    t_init: torch.Tensor,
    robust_loss: str = "cauchy_mad",
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gauss-Newton + IRLS for G-ICP and variants (p2p, p2pl, pl2pl, covariance).
    """
    R = R_init.clone()
    t = t_init.clone()
    N = source_points.shape[0]
    device = source_points.device

    def skew(v: torch.Tensor) -> torch.Tensor:  # encode cross product
        return torch.tensor(
            [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
            device=v.device,
        )

    Xr = (R @ source_points.T).T + t
    tgt = target_points[correspondences]
    diff = Xr - tgt

    L_i = L_target
    L_sqrt = torch.linalg.cholesky(L_i)  # (N, 3, 3)
    e = torch.einsum("nij,nj->ni", L_sqrt, diff)

    res = torch.linalg.norm(e, dim=1)
    w = _robust_loss_weights(res, robust_loss, **kwargs)

    Xr_rot = (R @ source_points.T).T

    S_Xr_rot = torch.zeros(N, 3, 3, device=device)
    S_Xr_rot[:, 0, 1] = -Xr_rot[:, 2]
    S_Xr_rot[:, 0, 2] = Xr_rot[:, 1]
    S_Xr_rot[:, 1, 0] = Xr_rot[:, 2]
    S_Xr_rot[:, 1, 2] = -Xr_rot[:, 0]
    S_Xr_rot[:, 2, 0] = -Xr_rot[:, 1]
    S_Xr_rot[:, 2, 1] = Xr_rot[:, 0]

    I3_batch = torch.eye(3, device=device).unsqueeze(0).expand(N, -1, -1)
    J_unweighted = torch.cat([-S_Xr_rot, I3_batch], dim=2)  # (N, 3, 6)

    J = torch.bmm(L_sqrt, J_unweighted)  # (N, 3, 6)

    w_sqrt = torch.sqrt(w).view(-1, 1)
    Jw = J * w_sqrt.unsqueeze(2)
    ew = e * w_sqrt

    Jmat = Jw.reshape(-1, 6)
    rvec = ew.reshape(-1)
    H = Jmat.T @ Jmat
    g = -Jmat.T @ rvec

    mu = 1e-4
    H += mu * torch.eye(6, device=H.device)

    try:
        delta = torch.linalg.solve(H, g)
    except torch.linalg.LinAlgError:
        delta = torch.linalg.pinv(H) @ g

    dtheta = delta[:3]
    dt = delta[3:]
    dR = torch.matrix_exp(skew(dtheta))  # exp-map SO(3)
    R = dR @ R
    t = t + dt

    return R, t


def _weighted_procrustes(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    weights: torch.Tensor,
    correspondences: torch.Tensor,
    L_mats: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve weighted Procrustes problem in closed form."""
    total_weight = weights.sum()
    src_centroid = (weights[:, None] * source_points).sum(0) / total_weight
    tgt = target_points[correspondences]
    tgt_centroid = (weights[:, None] * tgt).sum(0) / total_weight

    src_centered = source_points - src_centroid
    tgt_centered = tgt - tgt_centroid

    # Compute weighted cross-covariance matrix H
    if L_mats is not None:
        Lt = torch.einsum("nij,nj->ni", L_mats, tgt_centered)
        H = torch.einsum("n,ni,nj->ij", weights, src_centered, Lt)
    else:
        H = torch.einsum("n,ni,nj->ij", weights, src_centered, tgt_centered)

    U, _, Vt = torch.linalg.svd(H)
    V = Vt.T
    R = V @ U.T
    if torch.det(R) < 0:
        V[:, -1] *= -1
        R = V @ U.T
    t = tgt_centroid - R @ src_centroid
    return R, t


def _init_pca_alignment(
    source_points: torch.Tensor, target_points: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialize using PCA alignment."""

    def compute_pca(points):
        centered = points - torch.mean(points, dim=0)
        U, S, Vt = torch.linalg.svd(centered.T @ centered)
        return U

    source_axes = compute_pca(source_points)
    target_axes = compute_pca(target_points)

    # Align
    R = torch.mm(target_axes, source_axes.T)

    # Ensure proper rotation
    if torch.det(R) < 0:
        source_axes[:, -1] *= -1
        R = torch.mm(target_axes, source_axes.T)

    source_centroid = torch.mean(source_points, dim=0)
    target_centroid = torch.mean(target_points, dim=0)
    t = target_centroid - torch.mv(R, source_centroid)

    return R, t


def _init_centroid_svd(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    init_corr: (
        np.ndarray
        | torch.Tensor
        | tuple[np.ndarray, np.ndarray]
        | tuple[torch.Tensor, torch.Tensor]
    ),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialize using centroid alignment + SVD with provided correspondences."""

    if (
        isinstance(init_corr, np.ndarray | torch.Tensor)
        and init_corr.shape[1] == 2
    ):
        idx = torch.as_tensor(
            init_corr, dtype=torch.long, device=source_points.device
        )
        src = source_points[idx[:, 0]]
        tgt = target_points[idx[:, 1]]

    elif isinstance(init_corr, tuple) and len(init_corr) == 2:
        src_arr, tgt_arr = init_corr
        src = torch.as_tensor(
            src_arr, dtype=torch.float32, device=source_points.device
        )
        tgt = torch.as_tensor(
            tgt_arr, dtype=torch.float32, device=source_points.device
        )
        assert src.shape == tgt.shape
        assert src.shape[1] == 3

    else:
        msg = (
            "init_corr must be a 2D array of indices or a tuple of two arrays."
        )
        raise ValueError(msg)

    weights = torch.ones(src.shape[0], device=src.device)
    return _weighted_procrustes(
        src,
        tgt,
        weights,
        correspondences=torch.arange(src.shape[0], device=src.device),
    )


def _init_centroid_shift(
    source_points: torch.Tensor, target_points: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialize with centroid shift only."""
    device = source_points.device
    R = torch.eye(3, device=device)
    t = torch.mean(target_points, dim=0) - torch.mean(source_points, dim=0)
    return R, t


def _init_fpfh(
    source_points: torch.Tensor, target_points: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialize using FPFH feature matching."""
    if not HAS_OPEN3D:
        msg = "open3d is not installed. Please install it for FPFH initialization."
        raise ImportError(msg)

    device = source_points.device
    src_np = source_points.cpu().numpy()
    tgt_np = target_points.cpu().numpy()

    source_o3d = o3d.geometry.PointCloud()
    source_o3d.points = o3d.utility.Vector3dVector(src_np)
    target_o3d = o3d.geometry.PointCloud()
    target_o3d.points = o3d.utility.Vector3dVector(tgt_np)

    mn = np.asarray(source_o3d.get_min_bound())
    mx = np.asarray(source_o3d.get_max_bound())
    diag = np.linalg.norm(mx - mn)  # diagonal length of the bounding box

    # ddaptive voxel down-sampling
    def downsample(pcd):
        n = len(pcd.points)
        if n <= 5000:
            return pcd
        elif n <= 100000:
            voxel_size = diag * 0.005
        elif n <= 1000000:
            voxel_size = diag * 0.01
        else:
            voxel_size = diag * 0.02
        return pcd.voxel_down_sample(voxel_size)

    source_down = downsample(source_o3d)
    target_down = downsample(target_o3d)

    # estimate normals for down-sampled point clouds
    def estimate_normals(pcd):
        k = min(30, max(10, int(len(pcd.points) / 1000)))
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(
                knn=min(k, len(pcd.points) - 1)
            )
        )

    estimate_normals(source_down)
    estimate_normals(target_down)

    # compute FPFH features
    fpfh_radius = diag * 0.05
    max_nn = 100
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=fpfh_radius, max_nn=min(max_nn, len(source_down.points) - 1)
        ),
    )
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=fpfh_radius, max_nn=min(max_nn, len(target_down.points) - 1)
        ),
    )

    # correspondence distance threshold
    distance_thr = diag * 0.05

    # adaptive RANSAC iterations based on down-sampled size
    max_iter = 500000
    criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(
        max_iteration=max_iter, confidence=0.9995
    )

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        mutual_filter=False,
        max_correspondence_distance=distance_thr,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            False
        ),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9
            ),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_thr
            ),
        ],
        criteria=criteria,
    )

    # extract rotation and translation
    trans = result.transformation
    R = torch.from_numpy(trans[:3, :3]).float().to(device)
    t = torch.from_numpy(trans[:3, 3]).float().to(device)

    return R, t


def _compute_local_covariance(
    points: torch.Tensor, k_neighbors: int = 20
) -> torch.Tensor:
    """
    Compute per-point covariance matrices
    """
    device = points.device
    n_points = points.shape[0]

    edge_index = knn_graph(points, k=k_neighbors, loop=False)
    _, nbr_idx = edge_index
    nbr_idx = nbr_idx.view(n_points, k_neighbors)

    covariances = torch.zeros(n_points, 3, 3, device=device)

    for i in range(n_points):
        neigh = points[nbr_idx[i]]
        mu = neigh.mean(dim=0, keepdim=True)
        cen = neigh - mu
        cov = cen.T @ cen / (k_neighbors - 1)
        cov += 1e-2 * torch.eye(3, device=device)
        covariances[i] = cov

    return covariances


def closest_point_coupling(
    source_points: torch.Tensor,
    tree: KDTree,
    k: int = 10,
    device: torch.device | None = None,
) -> torch.Tensor:
    # find correspondences in Euclidean space
    _, idxs = tree.query(source_points, k=k)
    return (
        torch.from_numpy(idxs[:, 0]).long().to(device)
    )  # corr is defined as indices in target


def rigid_icp(
    source_shape: torch.Tensor | Any,
    target_shape: torch.Tensor | Any,
    initialization: str = "centroid_shift",
    R0: torch.Tensor | None = None,
    t0: torch.Tensor | None = None,
    robust_loss: str = "cauchy_mad",
    metric_type: str = "point_to_point",
    max_iterations: int = 50,
    tolerance: float = 1e-4,
    device: torch.device | None = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    """
    Rigid ICP
    """

    src_pts, src_nml = _extract_points_and_normals(source_shape)
    tgt_pts, tgt_nml = _extract_points_and_normals(target_shape)
    device = device or src_pts.device
    src = src_pts.to(device)
    tgt = tgt_pts.to(device)

    if src_nml is not None:
        src_nml = src_nml.to(device)
    if tgt_nml is not None:
        tgt_nml = tgt_nml.to(device)

    if metric_type in ["point_to_plane", "plane_to_plane"] and tgt_nml is None:
        msg = f"Target normals are required for metric_type='{metric_type}' but not provided"
        raise ValueError(msg)
    if metric_type == "plane_to_plane" and src_nml is None:
        msg = "Source normals are required for metric_type='plane_to_plane' but not provided"
        raise ValueError(msg)

    # build KD-tree once
    tree = KDTree(tgt.cpu().numpy(), leaf_size=40, metric="euclidean")

    # initialize R, t
    if R0 is not None and t0 is not None:
        R, t = R0.to(device), t0.to(device)
    elif initialization == "pca":
        R, t = _init_pca_alignment(src, tgt)
    elif initialization == "centroid_svd":
        R, t = _init_centroid_svd(src, tgt, kwargs.get("init_corr"))
    elif initialization == "fpfh":
        R, t = _init_fpfh(src, tgt)
    else:
        R, t = _init_centroid_shift(src, tgt)
    R_init, t_init = R.clone(), t.clone()

    # precompute covariances only for local_covariance metric
    if metric_type == "local_covariance":
        k_neighbors = kwargs.get("k_neighbors", 20)
        cov_src = _compute_local_covariance(src, k_neighbors)  # (N,3,3)
        cov_tgt = _compute_local_covariance(tgt, k_neighbors)  # (M,3,3)
    else:
        cov_src = cov_tgt = None

    prev_cost = float("inf")
    costs = []
    converged = False
    k = kwargs.get("k_candidates", 10)
    trans_tol = kwargs.get("trans_tol", 1e-2)
    angle_tol = kwargs.get("angle_tol", 1e-2)
    alpha = kwargs.get("alpha", 100.0)
    beta = kwargs.get("beta", 1.0)

    for _ in range(1, max_iterations + 1):
        # find correspondences in Euclidean space
        pts_trans = (src @ R.T + t).cpu().numpy()
        corr = closest_point_coupling(
            pts_trans,
            tree,
            k=k,
            device=device,
        )

        # compute metric matrices based on metric type
        if metric_type == "point_to_point":
            # simple Euclidean distance
            tgt_corr = tgt[corr]
            diff = (src @ R.T + t) - tgt_corr
            res2 = torch.sum(diff * diff, dim=1)
            L_i = None

        elif metric_type == "point_to_plane":
            # use target normals only
            tgt_corr_nml = tgt_nml[corr] if tgt_nml is not None else None
            L_i = _compute_local_metric(
                tgt[corr],
                tgt_corr_nml,
                metric_type="point_to_plane",
                alpha=alpha,
                beta=beta,
            )
            tgt_corr = tgt[corr]
            diff = (src @ R.T + t) - tgt_corr
            res2 = torch.einsum("ni,nij,nj->n", diff, L_i, diff)

        elif metric_type == "plane_to_plane":
            # use both source and target normals
            src_nml_trans = src_nml @ R.T if src_nml is not None else None
            tgt_corr_nml = tgt_nml[corr] if tgt_nml is not None else None
            L_i = _compute_local_metric(
                tgt[corr],
                tgt_corr_nml,
                metric_type="plane_to_plane",
                alpha=alpha,
                beta=beta,
                source_normals=src_nml_trans,
            )
            tgt_corr = tgt[corr]
            diff = (src @ R.T + t) - tgt_corr
            res2 = torch.einsum("ni,nij,nj->n", diff, L_i, diff)

        elif metric_type == "local_covariance":
            # rotate source covariance by current R
            rotcov = torch.einsum("ij,njk,kl->nil", R, cov_src, R.T)
            # build per-match sum and invert
            cov_sum = rotcov + cov_tgt[corr]
            L_i = torch.linalg.inv(cov_sum)
            tgt_corr = tgt[corr]
            diff = (src @ R.T + t) - tgt_corr
            res2 = torch.einsum("ni,nij,nj->n", diff, L_i, diff)

        residuals = torch.sqrt(res2)
        weights = _robust_loss_weights(residuals, robust_loss, **kwargs)

        # one update step
        if metric_type == "point_to_point":
            R_new, t_new = _weighted_procrustes(
                src,
                tgt,
                weights,
                corr,
                L_mats=None,
            )
        else:
            R_new, t_new = _gauss_newton_gicp(
                src, tgt, corr, L_i, R, t, robust_loss=robust_loss, **kwargs
            )

        # convergence test
        cost = (weights * res2).sum()
        trans_delta = torch.norm(t_new - t)
        R_rel = R_new @ R.T
        angle = torch.acos(
            torch.clamp((torch.trace(R_rel) - 1) / 2, -1 + 1e-7, 1 - 1e-7)
        )
        cost_delta = abs(prev_cost - cost.item())

        R, t = R_new, t_new
        prev_cost = cost.item()
        costs.append(prev_cost)

        if (
            trans_delta < trans_tol
            and angle < angle_tol
            and cost_delta < tolerance
        ):
            converged = True
            break

    info = {
        "final_cost": prev_cost,
        "costs": costs,
        "converged": converged,
        "last_trans_delta": trans_delta,
        "last_rot_angle": angle,
        "last_cost_delta": cost_delta,
    }
    return R, t, info, R_init, t_init
