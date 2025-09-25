from typing import Any

import numpy as np
import torch
from sklearn.neighbors import KDTree

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
        eps = 1e-8
        source_normals_norm = source_normals / (
            torch.norm(source_normals, dim=1, keepdim=True) + eps
        )
        target_normals_norm = target_normals / (
            torch.norm(target_normals, dim=1, keepdim=True) + eps
        )

        source_nn_T = torch.bmm(
            source_normals_norm.unsqueeze(2), source_normals_norm.unsqueeze(1)
        )

        target_nn_T = torch.bmm(
            target_normals_norm.unsqueeze(2), target_normals_norm.unsqueeze(1)
        )

        L_combined = alpha * (source_nn_T + target_nn_T)
        cross_prod = torch.cross(
            source_normals_norm, target_normals_norm, dim=1
        )
        cross_prod_nn_T = torch.bmm(
            cross_prod.unsqueeze(2), cross_prod.unsqueeze(1)
        )

        L_combined += beta * cross_prod_nn_T

        L_combined += 1e-6 * torch.eye(3, device=device).unsqueeze(0)

        return L_combined

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


def _gauss_newton_gicp_with_scale(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    correspondences: torch.Tensor,
    L_target: torch.Tensor,
    R_init: torch.Tensor,
    t_init: torch.Tensor,
    s_init: float = 1.0,
    robust_loss: str = "cauchy_mad",
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Gauss-Newton + IRLS pour G-ICP point-to-plane avec estimation de l'échelle.
    Retourne (R, t, s).
    """
    R = R_init.clone()
    t = t_init.clone()
    s = torch.as_tensor(s_init, dtype=R.dtype).to(R.device)

    def skew(v):
        return torch.tensor(
            [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
            device=v.device,
        )

    N = source_points.shape[0]
    device = R.device

    for _ in range(kwargs.get("max_gn_iter", 10)):
        Xr = (s * (R @ source_points.T)).T + t  # (N,3)
        if correspondences.is_sparse:
            tgt = torch.sparse.mm(correspondences, target_points)
        else:
            tgt = torch.mm(correspondences, target_points)
        diff = Xr - tgt  # (N,3)

        eigvals, eigvecs = torch.linalg.eigh(L_target)
        eigvals_clamped = torch.clamp(eigvals, min=0)
        L_sqrt = torch.einsum(
            "...ij,...j,...kj->...ik",
            eigvecs,
            torch.sqrt(eigvals_clamped),
            eigvecs,
        )  # (N,3,3)
        e = torch.einsum("nij,nj->ni", L_sqrt, diff)  # (N,3)

        res = torch.linalg.norm(e, dim=1)
        w = _robust_loss_weights(res, robust_loss)
        w_sqrt = torch.sqrt(w).unsqueeze(1)  # (N,1)
        e_w = (e * w_sqrt).reshape(-1)  # (3N,)

        Xr_rot = (s * (R @ source_points.T)).T  # (N,3)
        S = torch.zeros(N, 3, 3, device=device)
        S[:, 0, 1] = -Xr_rot[:, 2]
        S[:, 0, 2] = Xr_rot[:, 1]
        S[:, 1, 0] = Xr_rot[:, 2]
        S[:, 1, 2] = -Xr_rot[:, 0]
        S[:, 2, 0] = -Xr_rot[:, 1]
        S[:, 2, 1] = Xr_rot[:, 0]
        I3 = torch.eye(3, device=device).unsqueeze(0).expand(N, 3, 3)

        Rx = (R @ source_points.T).T  # (N,3)
        J_unweighted = torch.cat(
            [-S, I3, Rx.unsqueeze(2)], dim=2  # (N,3,3)  # (N,3,3)  # (N,3,1)
        )  # -> (N,3,7)

        J = torch.einsum("nij,njk->nik", L_sqrt, J_unweighted)  # (N,3,7)
        Jw = (J * w_sqrt.unsqueeze(2)).reshape(-1, 7)  # (3N,7)

        H = Jw.T @ Jw  # (7,7)
        g = -Jw.T @ e_w  # (7,)

        mu = 1e-6
        H += mu * torch.eye(7, device=device)

        scale_reg = kwargs.get("scale_reg", 1e2)
        H[6, 6] += scale_reg

        try:
            delta = torch.linalg.solve(H, g)
        except torch.linalg.LinAlgError:
            delta = torch.linalg.pinv(H) @ g

        dtheta = delta[:3]
        dt_vec = delta[3:6]
        ds = delta[6]

        dR = torch.matrix_exp(skew(dtheta))
        R = dR @ R
        t = t + dt_vec
        s = s + ds

        if (
            torch.norm(dt_vec) < kwargs.get("trans_tol", 1e-3)
            and torch.abs(ds) < kwargs.get("scale_tol", 1e-4)
            and torch.acos(torch.clamp((torch.trace(dR) - 1) / 2, -1, 1))
            < kwargs.get("angle_tol", 1e-3)
        ):
            break

    return R, t, s


def _gauss_newton_gicp(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    correspondences: torch.Tensor,
    L_target: torch.Tensor,
    R_init: torch.Tensor,
    t_init: torch.Tensor,
    robust_loss: str = "cauchy_mad",
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
    if correspondences.is_sparse:
        tgt = torch.sparse.mm(correspondences, target_points)
    else:
        tgt = torch.mm(correspondences, target_points)
    diff = Xr - tgt

    L_i = L_target
    eigvals, eigvecs = torch.linalg.eigh(L_i)  # (N,3), (N,3,3)
    eigvals = torch.clamp(eigvals, min=0.0)  # drop negatives
    L_sqrt = torch.einsum(
        "nij,nj,nkj->nik", eigvecs, torch.sqrt(eigvals), eigvecs
    )  # (N,3,3)
    e = torch.einsum("nij,nj->ni", L_sqrt, diff)

    res = torch.linalg.norm(e, dim=1)
    w = _robust_loss_weights(res, robust_loss)

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

    mu = 1e-6
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


def _weighted_similarity_procrustes(
    source_points: torch.Tensor,  # (N,3)
    target_points: torch.Tensor,  # (N,3) matched to source
    weights: torch.Tensor,  # (N,)
    correspondences: torch.Tensor,  # (N,) indices into target_points
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Solve weighted similarity (scale + rotation + translation) Procrustes:
    returns (R, t, s).
    """
    if correspondences.is_sparse:
        tgt = torch.sparse.mm(correspondences, target_points)
    else:
        tgt = torch.mm(correspondences, target_points)
    W = weights.sum()

    src_centroid = (weights[:, None] * source_points).sum(0) / W
    tgt_centroid = (weights[:, None] * tgt).sum(0) / W

    Xc = source_points - src_centroid  # (N,3)
    Yc = tgt - tgt_centroid  # (N,3)

    H = torch.einsum("n,ni,nj->ij", weights, Xc, Yc)

    U, S, Vt = torch.linalg.svd(H)
    V = Vt.T
    D = torch.diag(
        torch.tensor([1.0, 1.0, torch.det(V @ U.T)], device=H.device)
    )
    R = V @ D @ U.T

    numerator = S.sum()
    denominator = torch.einsum("n,ni->", weights, Xc.pow(2))
    s = numerator / denominator

    t = tgt_centroid - s * (R @ src_centroid)

    return R, t, s


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
    if correspondences.is_sparse:
        tgt = torch.sparse.mm(correspondences, target_points)
    else:
        tgt = torch.mm(correspondences, target_points)
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


def _init_pca_alignment_scale(
    source_points: torch.Tensor, target_points: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialize using PCA alignment."""

    def compute_pca(points):
        centered = points - torch.mean(points, dim=0)
        U, S, Vt = torch.linalg.svd(centered.T @ centered)
        return U

    source_axes = compute_pca(source_points)
    target_axes = compute_pca(target_points)

    R = target_axes @ source_axes.T
    if torch.det(R) < 0:
        source_axes[:, -1] *= -1
        R = target_axes @ source_axes.T

    mu_s = source_points.mean(0)
    mu_t = target_points.mean(0)

    src_proj = (source_points - mu_s) @ source_axes  # (N,3)
    tgt_proj = (target_points - mu_t) @ target_axes  # (M,3)
    var_s = torch.sum(src_proj.pow(2)) / source_points.shape[0]
    var_t = torch.sum(tgt_proj.pow(2)) / target_points.shape[0]
    scale = torch.sqrt(var_t / var_s)

    t = mu_t - scale * (R @ mu_s)
    return R, t, scale


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


def _init_centroid_svd_scale(src, tgt):
    mu_s = src.mean(0)
    mu_t = tgt.mean(0)
    Xc = src - mu_s
    Yc = tgt - mu_t

    H = Xc.T @ Yc
    U, S, Vt = torch.linalg.svd(H)
    V = Vt.T
    D = torch.diag(torch.tensor([1, 1, torch.det(V @ U.T)], device=H.device))
    R = V @ D @ U.T

    s = S.sum() / (Xc.pow(2).sum())
    t = mu_t - s * (R @ mu_s)
    return R, t, s


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


def _init_centroid_shift_scale(src, tgt):
    """Initialize transformation with centroid shift and scale from bounding box diagonal."""
    mu_s = src.mean(0)
    mu_t = tgt.mean(0)

    # Compute scale based on bounding box diagonals
    min_s = src.min(0)[0]
    max_s = src.max(0)[0]
    min_t = tgt.min(0)[0]
    max_t = tgt.max(0)[0]

    diag_s = torch.norm(max_s - min_s)
    diag_t = torch.norm(max_t - min_t)
    scale = diag_t / diag_s

    R = torch.eye(3, device=src.device)
    t = mu_t - scale * (R @ mu_s)
    return R, t, scale


def _init_centroid_shift(
    source_points: torch.Tensor, target_points: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialize with centroid shift only."""
    device = source_points.device
    R = torch.eye(3, device=device)
    t = torch.mean(target_points, dim=0) - torch.mean(source_points, dim=0)
    return R, t


def _init_fpfh_scale(
    source_points: torch.Tensor, target_points: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialize using FPFH feature matching."""
    if not HAS_OPEN3D:
        msg = "open3d is not installed. Please install it for FPFH initialization."
        raise ImportError(msg)

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

    # extract rotation and translation and scale
    trans = result.transformation
    M = trans[:3, :3]
    U, Svals, Vt = np.linalg.svd(M)
    R = torch.from_numpy(U @ Vt).float()
    scale = Svals.mean()
    t = torch.from_numpy(trans[:3, 3]).float()
    return R, t, torch.tensor(scale, dtype=torch.float32)


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

    # Use KDTree instead of knn_graph
    tree = KDTree(points.cpu().numpy(), leaf_size=40, metric="euclidean")
    _, nbr_indices = tree.query(
        points.cpu().numpy(), k=k_neighbors + 1
    )  # +1 because it includes the point itself
    nbr_idx = (
        torch.from_numpy(nbr_indices[:, 1:]).long().to(device)
    )  # exclude the point itself

    covariances = torch.zeros(n_points, 3, 3, device=device)

    for i in range(n_points):
        neigh = points[nbr_idx[i]]
        mu = neigh.mean(dim=0, keepdim=True)
        cen = neigh - mu
        cov = cen.T @ cen / (k_neighbors - 1)
        cov += 1e-2 * torch.eye(3, device=device)
        covariances[i] = cov

    return covariances


class ClosestPointCoupling:
    def __init__(self, device: torch.device | None = None):
        self.device = device

    def initialize(
        self, target: torch.Tensor, leaf_size: int = 40
    ) -> "ClosestPointCoupling":
        self.target = target.to(self.device)
        self.tree = KDTree(
            target.cpu().numpy(), leaf_size=leaf_size, metric="euclidean"
        )
        self._M = target.shape[0]

        return self

    def fit(self, source: torch.Tensor) -> "ClosestPointCoupling":
        self._N = source.shape[0]
        # find correspondences in Euclidean space
        _, idxs = self.tree.query(source.detach().numpy(), k=1)
        self._corr = (
            torch.from_numpy(idxs[:, 0]).long().to(self.device)
        )  # corr is defined as indices in target
        return self

    def to_sparse(self) -> torch.Tensor:
        if not hasattr(self, "_corr"):
            msg = "The correspondence must be computed before converting to sparse."
            raise ValueError(msg)

        idx_i = torch.arange(self._N, device=self.device)
        idx_j = self._corr

        indices = torch.stack([idx_i, idx_j], dim=0)

        values = torch.ones(self._N, device=self.device)

        return torch.sparse_coo_tensor(
            indices, values, size=(self._N, self._M), device=self.device
        )

    def to_dense(self) -> torch.Tensor:
        """Warning: Can consume a lot of memory for large point clouds."""
        if not hasattr(self, "_corr"):
            msg = "The correspondence must be computed before converting to dense."
            raise ValueError(msg)

        dense_corr = torch.zeros(self._N, self._M, device=self.device)
        dense_corr[torch.arange(self._N, device=self.device), self._corr] = 1.0

        return dense_corr

    def transport(
        self,
        measure: torch.Tensor,
    ) -> torch.Tensor:
        """Transport a discrete measure vector along the coupling."""

        return torch.sparse.mm(
            self.to_sparse().transpose(0, 1), measure.view(-1, 1)
        ).view(-1)

    def transfer(
        self,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Transfer a field (e.g. colors, textures) defined on one point set to the other."""

        return torch.sparse.mm(self.to_sparse(), values)

    def transform(self) -> torch.Tensor:
        """Transform the source point cloud using the closest point coupling."""
        if not hasattr(self, "_corr"):
            msg = "The correspondence must be computed before transforming."
            raise ValueError(msg)

        return self.target[self._corr].to(self.device)


class RigidDeformation:
    def __init__(
        self,
        initialization: str = "centroid_shift",
        robust_loss: str = "cauchy_mad",
        metric_type: str = "point_to_point",
        max_iterations: int = 50,
        tolerance: float = 1e-4,
        scale: bool = True,
        coupling: ClosestPointCoupling | None = None,
        R0: torch.Tensor | None = None,
        t0: torch.Tensor | None = None,
        s0: float | None = None,
        device: torch.device | None = None,
        **kwargs,
    ):
        self.initialization = initialization
        self.robust_loss = robust_loss
        self.metric_type = metric_type
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.scale = scale
        self.coupling = coupling
        self.R0 = R0
        self.t0 = t0
        self.s0 = s0
        self.device = device or torch.device("cpu")
        self.kwargs = kwargs

    def initialize(
        self, source_shape: torch.Tensor, target_shape: torch.Tensor
    ):
        self.source_shape = source_shape
        self.target_shape = target_shape

        self.src_pts, self.src_nml = _extract_points_and_normals(source_shape)
        self.tgt_pts, self.tgt_nml = _extract_points_and_normals(target_shape)
        device = self.device or self.src_pts.device
        self.src = self.src_pts.to(device)
        self.tgt = self.tgt_pts.to(device)

        # initialize R, t
        if not self.scale:
            if self.R0 is not None and self.t0 is not None:
                self.R, self.t = self.R0.to(self.device), self.t0.to(
                    self.device
                )
            elif self.initialization == "pca":
                self.R, self.t = _init_pca_alignment(self.src, self.tgt)
            elif self.initialization == "centroid_svd":
                self.R, self.t = _init_centroid_svd(
                    self.src, self.tgt, self.kwargs.get("init_corr")
                )
            elif self.initialization == "fpfh":
                self.R, self.t = _init_fpfh(self.src, self.tgt)
            else:
                self.R, self.t = _init_centroid_shift(self.src, self.tgt)
            self.R_init, self.t_init = self.R.clone(), self.t.clone()
            self.s_global = torch.tensor(1.0, device=device)
        if self.scale:
            if (
                self.R0 is not None
                and self.t0 is not None
                and self.s0 is not None
            ):
                self.R, self.t, self.s = (
                    self.R0.to(device),
                    self.t0.to(device),
                    self.s0,
                )
            elif self.initialization == "pca":
                self.R, self.t, self.s = _init_pca_alignment_scale(
                    self.src, self.tgt
                )
            elif self.initialization == "centroid_svd":
                self.R, self.t, self.s = _init_centroid_svd_scale(
                    self.src, self.tgt
                )
            elif self.initialization == "fpfh":
                self.R, self.t, self.s = _init_fpfh_scale(self.src, self.tgt)
            else:
                self.R, self.t, self.s = _init_centroid_shift_scale(
                    self.src, self.tgt
                )
            self.R_init, self.t_init = self.R.clone(), self.t.clone()
            self.s_global = self.s
            self.s_init = self.s_global

        # precompute covariances only for local_covariance metric
        if self.metric_type == "local_covariance":
            k_neighbors = self.kwargs.get("k_neighbors", 20)
            self.cov_src = _compute_local_covariance(
                self.src, k_neighbors
            )  # (N,3,3)
            self.cov_tgt = _compute_local_covariance(
                self.tgt, k_neighbors
            )  # (M,3,3)
        else:
            self.cov_src = self.cov_tgt = None

    def _compute_metric(
        self,
        corr: torch.Tensor | torch.sparse.Tensor,
    ):
        alpha = self.kwargs.get("alpha", 100.0)
        beta = self.kwargs.get("beta", 1.0)
        # compute metric matrices based on metric type
        if self.metric_type == "point_to_point":
            # simple Euclidean distance
            if corr.is_sparse:
                tgt_corr = torch.sparse.mm(corr, self.tgt)
            else:
                tgt_corr = torch.mm(corr, self.tgt)
            diff = (self.s_global * (self.src @ self.R.T) + self.t) - tgt_corr
            self.res2 = torch.sum(diff * diff, dim=1)
            self.L_i = None

        elif self.metric_type == "point_to_plane":
            # use target normals only
            if corr.is_sparse:
                tgt_corr_nml = (
                    torch.sparse.mm(corr, self.tgt_nml)
                    if self.tgt_nml is not None
                    else None
                )
                tgt_corr = torch.sparse.mm(corr, self.tgt)
            else:
                tgt_corr_nml = (
                    torch.mm(corr, self.tgt_nml)
                    if self.tgt_nml is not None
                    else None
                )
                tgt_corr = torch.mm(corr, self.tgt)
            self.L_i = _compute_local_metric(
                tgt_corr,
                tgt_corr_nml,
                metric_type="point_to_plane",
                alpha=alpha,
                beta=beta,
            )
            diff = (self.s_global * (self.src @ self.R.T) + self.t) - tgt_corr
            self.res2 = torch.einsum("ni,nij,nj->n", diff, self.L_i, diff)

        elif self.metric_type == "plane_to_plane":
            # use both source and target normals
            src_nml_trans = (
                self.src_nml @ self.R.T if self.src_nml is not None else None
            )
            if corr.is_sparse:
                tgt_corr_nml = (
                    torch.sparse.mm(corr, self.tgt_nml)
                    if self.tgt_nml is not None
                    else None
                )
                tgt_corr = torch.sparse.mm(corr, self.tgt)
            else:
                tgt_corr_nml = (
                    torch.mm(corr, self.tgt_nml)
                    if self.tgt_nml is not None
                    else None
                )
                tgt_corr = torch.mm(corr, self.tgt)
            self.L_i = _compute_local_metric(
                tgt_corr,
                tgt_corr_nml,
                metric_type="plane_to_plane",
                alpha=alpha,
                beta=beta,
                source_normals=src_nml_trans,
            )
            diff = (self.s_global * (self.src @ self.R.T) + self.t) - tgt_corr
            self.res2 = torch.einsum("ni,nij,nj->n", diff, self.L_i, diff)

        elif self.metric_type == "local_covariance":
            # rotate source covariance by current R
            rotcov = (self.s_global**2) * torch.einsum(
                "ij,njk,kl->nil", self.R, self.cov_src, self.R.T
            )
            M = self.cov_tgt.shape[0]
            cov_tgt_flat = self.cov_tgt.reshape(M, 9)
            if corr.is_sparse:
                cov_tgt_corr_flat = torch.sparse.mm(corr, cov_tgt_flat)
                tgt_corr = torch.sparse.mm(corr, self.tgt)
            else:
                cov_tgt_corr_flat = torch.mm(corr, cov_tgt_flat)
                tgt_corr = torch.mm(corr, self.tgt)
            cov_tgt_corr = cov_tgt_corr_flat.reshape(-1, 3, 3)
            # build per-match sum and invert
            cov_sum = rotcov + cov_tgt_corr
            self.L_i = torch.linalg.inv(cov_sum)
            diff = (self.s_global * (self.src @ self.R.T) + self.t) - tgt_corr
            self.res2 = torch.einsum("ni,nij,nj->n", diff, self.L_i, diff)

        self.residuals = torch.sqrt(self.res2)
        self.weights = _robust_loss_weights(self.residuals, self.robust_loss)

    def optimization_step(self, coupling: ClosestPointCoupling):
        self.corr = coupling.to_dense()
        self._compute_metric(self.corr)

        # one update step
        if not self.scale:
            if self.metric_type == "point_to_point":
                self.R, self.t = _weighted_procrustes(
                    self.src,
                    self.tgt,
                    self.weights,
                    self.corr,
                    L_mats=None,
                )
            else:  # gauss_newton
                self.R, self.t = _gauss_newton_gicp(
                    self.src,
                    self.tgt,
                    self.corr,
                    self.L_i,
                    self.R,
                    self.t,
                    robust_loss=self.robust_loss,
                    **self.kwargs,
                )
            self.s = self.s_global
        elif self.metric_type == "point_to_point":
            self.R, self.t, self.s = _weighted_similarity_procrustes(
                self.src,
                self.tgt,
                self.weights,
                self.corr,
            )
        else:  # gauss_newton
            self.R, self.t, self.s = _gauss_newton_gicp_with_scale(
                self.src,
                self.tgt,
                self.corr,
                self.L_i,
                self.R,
                self.t,
                s_init=self.s_global,
                robust_loss=self.robust_loss,
                **self.kwargs,
            )

        self.s_global = self.s
        self.rotation_ = self.R
        self.translation_ = self.t
        if self.scale:
            self.scale_ = self.s

        self.R = self.R.detach()
        self.t = self.t.detach()
        if self.scale:
            self.s = self.s.detach()

    def transform(self):
        # Transform the source shape
        if HAS_PYVISTA and isinstance(self.source_shape, pv.PolyData):
            transformed_shape = self.source_shape.copy()
            transformed_points = (
                (self.s_global * (self.src @ self.R.T) + self.t).cpu().numpy()
            )
            transformed_shape.points = transformed_points

            if self.src_nml is not None:
                transformed_normals = (self.src_nml @ self.R.T).cpu().numpy()
                if "Normals" in transformed_shape.point_data:
                    transformed_shape.point_data["Normals"] = (
                        transformed_normals
                    )
        elif isinstance(self.source_shape, torch.Tensor | np.ndarray):
            transformed_points = (
                (self.s_global * (self.src @ self.R.T) + self.t).cpu().numpy()
            )
            if isinstance(self.source_shape, torch.Tensor):
                transformed_shape = torch.from_numpy(transformed_points).to(
                    self.source_shape.device
                )
            else:
                transformed_shape = transformed_points
        elif (
            isinstance(self.source_shape, tuple)
            and len(self.source_shape) == 2
        ):
            transformed_points = (
                (self.s_global * (self.src @ self.R.T) + self.t).cpu().numpy()
            )
            if self.src_nml is not None:
                transformed_normals = (self.src_nml @ self.R.T).cpu().numpy()
            else:
                transformed_normals = self.source_shape[1]

            if isinstance(self.source_shape[0], torch.Tensor):
                transformed_points = torch.from_numpy(transformed_points).to(
                    self.source_shape[0].device
                )
                if (
                    isinstance(self.source_shape[1], torch.Tensor)
                    and transformed_normals is not None
                ):
                    transformed_normals = torch.from_numpy(
                        transformed_normals
                    ).to(self.source_shape[1].device)

            transformed_shape = (transformed_points, transformed_normals)
        else:
            transformed_shape = (
                (self.s_global * (self.src @ self.R.T) + self.t).cpu().numpy()
            )

        return transformed_shape

    def fit(self):
        prev_cost = float("inf")
        costs = []
        converged = False
        trans_tol = self.kwargs.get("trans_tol", 1e-2)
        angle_tol = self.kwargs.get("angle_tol", 1e-2)

        for _ in range(1, self.max_iterations + 1):
            t = self.t
            R = self.R
            if self.scale:
                s_global = self.s_global
            else:
                s_global = torch.tensor(1.0, device=self.src.device)

            # find correspondences in Euclidean space
            pts_trans = s_global * self.src @ R.T + t

            self.coupling.fit(source=pts_trans)

            self.optimization_step(self.coupling)

            # convergence test
            cost = (self.weights * self.res2).sum()
            trans_delta = torch.norm(self.t - t)
            R_rel = self.R @ R.T
            angle = torch.acos(
                torch.clamp((torch.trace(R_rel) - 1) / 2, -1 + 1e-7, 1 - 1e-7)
            )
            cost_delta = abs(prev_cost - cost.item())
            s_delta = abs(self.s_global - s_global)

            prev_cost = cost.item()
            costs.append(prev_cost)

            if (
                trans_delta < trans_tol
                and angle < angle_tol
                and cost_delta < self.tolerance
                and s_delta < self.kwargs.get("scale_tol", 1e-4)
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

        if not self.scale:
            return (
                self.R,
                self.t,
                info,
                self.R_init,
                self.t_init,
            )
        else:
            return (
                self.R,
                self.t,
                self.s_global,
                info,
                self.R_init,
                self.t_init,
                self.s_init,
            )

    @property
    def transformation_matrix(self) -> torch.Tensor:
        # Check if the transformation parameters have been initialized
        if not all(hasattr(self, attr) for attr in ["R", "t", "s_global"]):
            msg = "Transformation not initialized. Call the 'initialize' or 'fit' method before accessing the matrix."
            raise AttributeError(msg)

        matrix = torch.eye(4, device=self.device, dtype=self.R.dtype)
        matrix[:3, :3] = self.s_global * self.R
        matrix[:3, 3] = self.t

        return matrix


def rigid_icp(
    source_shape: torch.Tensor | Any,
    target_shape: torch.Tensor | Any,
    initialization: str = "centroid_shift",
    R0: torch.Tensor | None = None,
    t0: torch.Tensor | None = None,
    s0: float | None = None,
    robust_loss: str = "cauchy_mad",
    metric_type: str = "point_to_point",
    max_iterations: int = 50,
    tolerance: float = 1e-4,
    scale: bool = True,
    device: torch.device | None = None,
    **kwargs,
) -> (
    tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
        torch.Tensor,
        torch.Tensor,
        Any,
    ]
    | tuple[
        torch.Tensor,
        torch.Tensor,
        float,
        dict[str, Any],
        torch.Tensor,
        torch.Tensor,
        float,
        Any,
    ]
):
    """
    Rigid ICP
    """

    tgt_pts, _ = _extract_points_and_normals(target_shape)

    # build KD-tree once
    coupling = ClosestPointCoupling(device=device)
    coupling.initialize(target=tgt_pts, leaf_size=kwargs.get("leaf_size", 40))

    model = RigidDeformation(
        initialization=initialization,
        robust_loss=robust_loss,
        metric_type=metric_type,
        max_iterations=max_iterations,
        tolerance=tolerance,
        scale=scale,
        coupling=coupling,
        R0=R0,
        t0=t0,
        s0=s0,
        device=device,
        **kwargs,
    )

    model.initialize(source_shape, target_shape)

    result = model.fit()
    transformed_shape = model.transform()

    if scale:
        (
            R,
            t,
            s_global,
            info,
            R_init,
            t_init,
            s_init,
        ) = result
        return (
            R,
            t,
            s_global,
            info,
            R_init,
            t_init,
            s_init,
            transformed_shape,
        )
    else:
        (
            R,
            t,
            info,
            R_init,
            t_init,
        ) = result
        return (R, t, info, R_init, t_init, transformed_shape)
