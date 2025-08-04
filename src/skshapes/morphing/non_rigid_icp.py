import logging
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
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

logger = logging.getLogger(__name__)


def _incidence_stiffness(
    edges: torch.Tensor,
    gamma: float,
    device: torch.device,
    alpha: float,
    edge_length: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d = 12
    E = edges.shape[0]
    i, j = edges.t()

    edge_rows = torch.arange(d, device=device).repeat(E)
    i_base = i.repeat_interleave(d)
    j_base = j.repeat_interleave(d)

    rows = torch.cat(
        [
            i_base * d + edge_rows,
            i_base * d + edge_rows,
            j_base * d + edge_rows,
            j_base * d + edge_rows,
        ]
    )

    cols = torch.cat(
        [
            i_base * d + edge_rows,
            j_base * d + edge_rows,
            i_base * d + edge_rows,
            j_base * d + edge_rows,
        ]
    )

    w_factor = torch.ones(d, device=device)
    w_factor[3::4] = gamma

    edge_length_array = edge_length.view(-1, 1)
    logger.debug(
        "Edge length min/max: %s, %s",
        torch.min(edge_length_array),
        torch.max(edge_length_array),
    )
    invL = (1.0 / edge_length).repeat_interleave(d, dim=0).reshape(-1)
    base = alpha * w_factor[edge_rows] * invL
    pattern = torch.tensor([1.0, -1.0, -1.0, 1.0], device=device)

    vals = (pattern.unsqueeze(1) * base).reshape(-1)

    return rows, cols, vals


def _solve_sparse_system(
    rows: torch.Tensor,
    cols: torch.Tensor,
    vals: torch.Tensor,
    Nd: int,
    b: torch.Tensor,
) -> torch.Tensor:
    A = sp.csr_matrix(
        (vals.cpu().numpy(), (rows.cpu().numpy(), cols.cpu().numpy())),
        shape=(Nd, Nd),
    )

    global_jitter = max(1e-4, Nd * 1e-12)
    A.setdiag(A.diagonal() + global_jitter)
    try:
        x_np = spla.spsolve(A, b.cpu().numpy())
    except Exception:
        x_np, info = spla.cg(
            A, b.cpu().numpy(), tol=1e-12, atol=0, maxiter=10 * Nd
        )
        if info != 0:
            logger.debug(
                "[warn] CG fallback failed to converge (info=%s)", info
            )

    return torch.as_tensor(x_np, device=b.device)


@torch.no_grad()
def _solve(
    source_pts,
    tgt,
    weights,
    corr,
    L_mats,
    edges,
    alpha_stiffness=1e2,
    gamma=1.0,
    rows_data: torch.Tensor = None,
    cols_data: torch.Tensor = None,
    stiffness_rows=None,
    stiffness_cols=None,
    stiffness_vals=None,
    device=None,
    residuals: torch.Tensor = None,
    metric_type: str = "point_to_point",
):

    d = 12
    N, _ = source_pts.shape
    device = source_pts.device if device is None else device
    w_factor = torch.ones(d, device=device)
    w_factor[3::4] = gamma
    Nd = N * d

    v_h = torch.cat([source_pts, torch.ones(N, 1, device=device)], 1)
    if L_mats is None:
        L_mats = torch.eye(3, device=device).expand(N, 3, 3)
    eps = 1e-12
    sqrt_w = torch.sqrt(torch.clamp(weights, min=eps))
    Wi = sqrt_w.view(-1, 1, 1) * L_mats

    Ai = torch.zeros(N, 3, d, device=device)
    for i, offset in enumerate([0, 4, 8]):
        Ai[:, i, offset : offset + 4] = v_h

    Ai_t = Ai.transpose(1, 2)
    Wi_Ai = torch.bmm(Wi, Ai)
    Mi = torch.bmm(Ai_t, Wi_Ai)

    lambda_block = (
        0.1 if metric_type in {"plane_to_plane", "point_to_plane"} else 0.01
    )

    eye12 = torch.eye(d, device=device).unsqueeze(0)
    Mi = Mi + lambda_block * eye12

    if residuals is None:
        tgt_corr = tgt[corr].unsqueeze(-1)
    else:
        tgt_corr = -residuals.unsqueeze(-1)

    Wi_tgt = torch.bmm(Wi, tgt_corr)
    bi = torch.bmm(Ai_t, Wi_tgt).squeeze(-1)

    vals_data = Mi.reshape(-1)

    rows = rows_data
    cols = cols_data
    vals = vals_data

    b = bi.reshape(-1)

    if stiffness_rows is not None:
        rows = torch.cat([rows, stiffness_rows])
        cols = torch.cat([cols, stiffness_cols])
        vals = torch.cat([vals, stiffness_vals])
    else:
        E = edges.shape[0]
        i_idx, j_idx = edges.t()
        edge_rows = torch.arange(d, device=device).repeat(E)
        i_base = i_idx.repeat_interleave(d)
        j_base = j_idx.repeat_interleave(d)

        rows = torch.cat(
            [
                rows,
                i_base * d + edge_rows,
                i_base * d + edge_rows,
                j_base * d + edge_rows,
                j_base * d + edge_rows,
            ]
        )
        cols = torch.cat(
            [
                cols,
                i_base * d + edge_rows,
                j_base * d + edge_rows,
                i_base * d + edge_rows,
                j_base * d + edge_rows,
            ]
        )
        vals = torch.cat(
            [
                vals,
                (
                    torch.tensor([1, -1, -1, 1], device=device).unsqueeze(1)
                    * (alpha_stiffness * w_factor[edge_rows])
                ).reshape(-1),
            ]
        )

    # b = torch.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)

    data_norm = torch.norm(vals_data, p="fro")
    stiff_norm = torch.norm(stiffness_vals, p="fro")
    logger.debug(
        " data-block norm = %.4e, stiffness-block norm = %.4e",
        data_norm,
        stiff_norm,
    )

    x = _solve_sparse_system(rows, cols, vals, Nd, b)
    # x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    delta = x.view(N, 3, 4)
    I3 = torch.eye(3, device=device).unsqueeze(0).expand(N, 3, 3)
    delta[:, :, :3] += I3
    X_delta = delta

    logger.debug("→ X_delta[0] rotation:\n%s", X_delta[0, :3, :3])
    logger.debug("→ X_delta[0] translation: %s", X_delta[0, :3, 3])

    return x.to(device).view(N, 3, 4)


def _extract_points_and_normals(
    shape: pv.PolyData | torch.Tensor | np.ndarray | tuple,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Extract points and normals from various input formats."""
    if HAS_PYVISTA and isinstance(shape, pv.PolyData):
        points = torch.from_numpy(shape.points).float()
        normals = None
        if "Normals" in shape.point_data:
            normals = torch.from_numpy(shape.point_data["Normals"]).float()
        elif (
            hasattr(shape, "point_normals") and shape.point_normals is not None
        ):
            normals = torch.from_numpy(shape.point_normals).float()
        return points, normals
    elif isinstance(shape, torch.Tensor | np.ndarray):
        if isinstance(shape, np.ndarray):
            shape = torch.from_numpy(shape).float()
        return (
            shape,
            None,
        )
    elif isinstance(shape, tuple) and len(shape) == 2:
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
        In33 = torch.eye(3, device=device).unsqueeze(0).repeat(n_points, 1, 1)
        target_normals = target_normals / torch.norm(
            target_normals, dim=1, keepdim=True
        )
        nn_T = torch.bmm(
            target_normals.unsqueeze(2), target_normals.unsqueeze(1)
        )
        eps = 1e-3
        return alpha * nn_T + beta * (In33 - nn_T) + eps * In33

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

        eps = 1e-3
        I3 = torch.eye(3, device=device).unsqueeze(0).expand(n_points, 3, 3)
        L_combined = alpha * (source_nn_T + target_nn_T) + eps * I3

        return L_combined  # noqa: RET504

    else:
        return torch.eye(3, device=device).unsqueeze(0).repeat(n_points, 1, 1)


def _robust_loss_weights(
    residuals: torch.Tensor, loss_type: str = "cauchy_mad", **kwargs
) -> torch.Tensor:
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

    elif loss_type == "welsch":
        k = kwargs.get("k", 1.0)
        return torch.exp(-((residuals / k) ** 2))

    else:
        return torch.ones_like(residuals)


def _compute_local_covariance(
    points: torch.Tensor, k_neighbors: int = 20
) -> torch.Tensor:
    device = points.device

    tree = KDTree(points.cpu().numpy(), leaf_size=40, metric="euclidean")
    _, nbr_idx = tree.query(points.cpu().numpy(), k=k_neighbors + 1)
    nbr_idx = nbr_idx[:, 1:]
    nbr_idx = torch.from_numpy(nbr_idx).to(device)

    nbr = points[nbr_idx]
    mu = nbr.mean(dim=1, keepdim=True)
    cen = nbr - mu

    cov = torch.matmul(cen.transpose(1, 2), cen) / (k_neighbors - 1)
    cov += 1e-2 * torch.eye(3, device=device).unsqueeze(0)

    return cov


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
        _, idxs = self.tree.query(source.cpu().numpy(), k=1)
        self._corr = torch.from_numpy(idxs[:, 0]).long().to(self.device)

        return self

    def to_sparse(self) -> torch.Tensor:
        if not hasattr(self, "corr"):
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


class NonRigidDeformation:
    def __init__(
        self,
        robust_loss: str = "cauchy_mad",
        metric_type: str = "point_to_point",
        max_iterations: int = 50,
        gamma: float = 1.0,
        device: torch.device | None = None,
        coupling: ClosestPointCoupling | None = None,
        alpha_start: float = 1e5,
        alpha_final: float = 1e1,
        n_levels: int = 6,
        **kwargs,
    ):
        self.robust_loss = robust_loss
        self.metric_type = metric_type
        self.max_iterations = max_iterations
        self.gamma = gamma
        self.device = device or torch.device("cpu")
        self.kwargs = kwargs

        self.alpha_start = alpha_start
        self.alpha_final = alpha_final
        self.n_levels = n_levels

        self.trans_tol = kwargs.get("trans_tol", 1e-2)
        self.angle_tol = kwargs.get("angle_tol", 1e-2)
        self.param_tol = kwargs.get("param_tol", 5e-4)
        self.cost_tol = kwargs.get("cost_tol", 1e-3)

        self.coupling = coupling

    def initialize(self, source_shape: Any, target_shape: Any):
        """Initialize the deformation with source and target shapes."""
        self.source_shape = source_shape
        self.target_shape = target_shape

        self.src_pts, self.src_nml = _extract_points_and_normals(source_shape)
        self.tgt_pts, self.tgt_nml = _extract_points_and_normals(target_shape)

        self.src = self.src_pts.to(self.device)
        self.tgt = self.tgt_pts.to(self.device)
        self.src_orig = self.src_pts.to(self.device)

        if self.src_nml is not None:
            self.src_nml = self.src_nml.to(self.device)
            self.src_nml_orig = self.src_nml.clone()
        else:
            self.src_nml_orig = None

        if self.tgt_nml is not None:
            self.tgt_nml = self.tgt_nml.to(self.device)

        if (
            self.metric_type in ["point_to_plane", "plane_to_plane"]
            and self.tgt_nml is None
        ):
            msg = f"Target normals are required for metric_type='{self.metric_type}' but not provided"
            raise ValueError(msg)
        if self.metric_type == "plane_to_plane" and self.src_nml is None:
            msg = "Source normals are required for metric_type='plane_to_plane' but not provided"
            raise ValueError(msg)

        self.n = self.src_orig.shape[0]
        self.d = 12

        self.X_total = (
            torch.eye(4, device=self.device).unsqueeze(0).repeat(self.n, 1, 1)
        )

        base = (torch.arange(self.n, device=self.device) * self.d).view(
            -1, 1, 1
        )
        rr, cc = torch.meshgrid(
            torch.arange(self.d, device=self.device),
            torch.arange(self.d, device=self.device),
            indexing="ij",
        )
        self.rows_data = (base + rr).reshape(-1)
        self.cols_data = (base + cc).reshape(-1)

        if self.coupling is None:
            self.tree = KDTree(
                self.tgt.cpu().numpy(), leaf_size=40, metric="euclidean"
            )
        else:
            self.tree = None

        if self.metric_type == "local_covariance":
            k_neighbors = self.kwargs.get("k_neighbors", 20)
            self.cov_src = _compute_local_covariance(self.src, k_neighbors)
            self.cov_tgt = _compute_local_covariance(self.tgt, k_neighbors)
        else:
            self.cov_src = self.cov_tgt = None

        self._setup_edges()

        self.alpha_levels = torch.logspace(
            torch.log10(torch.tensor(self.alpha_start)),
            torch.log10(torch.tensor(self.alpha_final)),
            steps=self.n_levels,
            device=self.device,
        )

    def _setup_edges(self):
        """Extract edges from the source mesh for stiffness regularization."""
        if HAS_PYVISTA and hasattr(self.source_shape, "extract_all_edges"):
            edge_mesh = self.source_shape.extract_all_edges()
            lines = edge_mesh.lines.reshape(-1, 3)
            edges_np = lines[:, 1:]

            edges_unique = np.unique(np.sort(edges_np, axis=1), axis=0)
            pts = self.src_orig.cpu().numpy()
            v0, v1 = pts[edges_unique[:, 0]], pts[edges_unique[:, 1]]
            lengths = np.linalg.norm(v0 - v1, axis=1, keepdims=True)

            # Filter out degenerate edges
            eps = 1e-6
            valid = lengths[:, 0] > eps
            edges_filtered = edges_unique[valid]
            lengths_filtered = lengths[valid]

            self.edges = torch.as_tensor(
                edges_filtered, dtype=torch.long, device=self.device
            )
            self.edge_length = (
                torch.from_numpy(lengths_filtered).float().to(self.device)
            )
        else:
            logger.warning(
                "No edge information available, using empty edge set"
            )
            self.edges = torch.empty(
                (0, 2), dtype=torch.long, device=self.device
            )
            self.edge_length = torch.empty(
                (0, 1), dtype=torch.float32, device=self.device
            )

    def _compute_correspondences(
        self, pts_trans: torch.Tensor
    ) -> torch.Tensor:
        """Find closest point correspondences."""
        if self.coupling is not None:
            # Use provided coupling
            return self.coupling.fit(pts_trans)._corr
        else:
            _, idxs = self.tree.query(pts_trans.cpu().numpy(), k=1)
            return torch.from_numpy(idxs[:, 0]).long().to(self.device)

    def _compute_metric(self, pts_trans: torch.Tensor, corr: torch.Tensor):
        """Compute residuals and metric matrices based on metric type."""
        alpha = self.kwargs.get("alpha", 100.0)
        beta = self.kwargs.get("beta", 1.0)

        if self.metric_type == "point_to_point":
            tgt_corr = self.tgt[corr]
            diff = pts_trans - tgt_corr
            res2 = torch.sum(diff * diff, dim=1)
            L_i = None

        elif self.metric_type == "point_to_plane":
            tgt_corr_nml = (
                self.tgt_nml[corr] if self.tgt_nml is not None else None
            )
            L_i = _compute_local_metric(
                self.tgt[corr],
                tgt_corr_nml,
                metric_type="point_to_plane",
                alpha=alpha,
                beta=beta,
            )
            tgt_corr = self.tgt[corr]
            diff = pts_trans - tgt_corr
            res2 = torch.einsum("ni,nij,nj->n", diff, L_i, diff)

        elif self.metric_type == "plane_to_plane":
            Rb = self.X_total[:, :3, :3]
            src_nml_trans = (
                torch.einsum("nij,nj->ni", Rb, self.src_nml_orig)
                if self.src_nml_orig is not None
                else None
            )
            tgt_corr_nml = (
                self.tgt_nml[corr] if self.tgt_nml is not None else None
            )
            L_i = _compute_local_metric(
                self.tgt[corr],
                tgt_corr_nml,
                metric_type="plane_to_plane",
                alpha=alpha,
                beta=beta,
                source_normals=src_nml_trans,
            )
            tgt_corr = self.tgt[corr]
            diff = pts_trans - tgt_corr
            res2 = torch.einsum("ni,nij,nj->n", diff, L_i, diff)

        elif self.metric_type == "local_covariance":
            Rb = self.X_total[:, :3, :3]
            rotcov = torch.bmm(Rb, torch.bmm(self.cov_src, Rb.transpose(1, 2)))
            cov_sum = rotcov + self.cov_tgt[corr]
            L_i = torch.linalg.inv(cov_sum)
            tgt_corr = self.tgt[corr]
            diff = pts_trans - tgt_corr
            res2 = torch.einsum("ni,nij,nj->n", diff, L_i, diff)

        return res2, L_i, diff

    def optimization_step(
        self, alpha_stiffness: float, stiffness_precomputed: tuple
    ):
        """Perform one optimization step."""
        stiffness_rows, stiffness_cols, stiffness_vals = stiffness_precomputed

        v_h = torch.cat(
            [self.src_orig, torch.ones(self.n, 1, device=self.device)], 1
        )
        pts_trans = torch.einsum("nij,nj->ni", self.X_total, v_h)[:, :3]

        corr = self._compute_correspondences(pts_trans)

        res2, L_i, diff = self._compute_metric(pts_trans, corr)

        residuals = torch.sqrt(res2)
        weights = _robust_loss_weights(
            residuals, self.robust_loss, **self.kwargs
        )
        wm = weights.mean()
        logger.debug("mean robust weight = %.2e", wm)
        weights /= torch.clamp(wm, min=1e-12)

        if torch.isnan(weights).any():
            msg = "NaN detected in robust weights"
            raise RuntimeError(msg)

        logger.debug(
            "weights min/max = %.2e/%.2e", weights.min(), weights.max()
        )

        source_linear = pts_trans.detach()
        X_delta = _solve(
            source_linear,
            self.tgt,
            weights,
            corr,
            L_mats=L_i,
            edges=self.edges,
            alpha_stiffness=alpha_stiffness,
            gamma=self.gamma,
            rows_data=self.rows_data,
            cols_data=self.cols_data,
            stiffness_rows=stiffness_rows,
            stiffness_cols=stiffness_cols,
            stiffness_vals=stiffness_vals,
            device=self.device,
            residuals=diff,
            metric_type=self.metric_type,
        )

        R_lin = X_delta[:, :, :3]
        U, S, Vt = torch.linalg.svd(R_lin)
        det = torch.linalg.det(U @ Vt)
        U_fix = U.clone()
        U_fix[det < 0, :, -1] *= -1
        R_proj = U_fix @ Vt
        X_delta[:, :, :3] = R_proj

        X_d_h = torch.zeros(self.n, 4, 4, device=self.device)
        X_d_h[:, :3, :4] = X_delta
        X_d_h[:, 3, 3] = 1.0
        X_total_prev = self.X_total.clone()

        self.X_total = torch.bmm(X_d_h, self.X_total)

        dets = torch.linalg.det(self.X_total[:, :3, :3])
        logger.debug("dets min/max = %.4f/%.4f", dets.min(), dets.max())

        Rb_new = self.X_total[:, :3, :3]
        Rb_old = X_total_prev[:, :3, :3]
        tb_new = self.X_total[:, :3, 3]
        tb_old = X_total_prev[:, :3, 3]

        dt = tb_new - tb_old
        trans_deltas = torch.norm(dt, dim=1)
        trans_delta = torch.max(trans_deltas)

        R_rel = torch.matmul(Rb_new, Rb_old.transpose(1, 2))
        traces = R_rel.diagonal(offset=0, dim1=1, dim2=2).sum(dim=1)
        eps = 1e-7
        angles = torch.acos(torch.clamp((traces - 1) / 2, -1 + eps, 1 - eps))
        angle = torch.max(angles)

        cost = (weights * res2).sum()
        delta_X = torch.norm((self.X_total - X_total_prev).view(-1))
        rel_par = delta_X / (torch.norm(X_total_prev.view(-1)) + 1e-12)

        return cost, trans_delta, angle, rel_par, corr

    def fit(self):
        """Run the complete non-rigid ICP optimization."""
        prev_cost = float("inf")
        costs = []
        converged = False

        for level, alpha_stiffness in enumerate(self.alpha_levels, 1):
            logger.debug(
                "\n=== alpha-level %s/%s : alpha = %.1e ===",
                level,
                self.n_levels,
                alpha_stiffness,
            )

            # Precompute stiffness matrix entries
            stiffness_rows, stiffness_cols, stiffness_vals = (
                _incidence_stiffness(
                    self.edges,
                    gamma=self.gamma,
                    device=self.device,
                    alpha=alpha_stiffness,
                    edge_length=self.edge_length,
                )
            )
            stiffness_precomputed = (
                stiffness_rows,
                stiffness_cols,
                stiffness_vals,
            )

            for it in range(1, self.max_iterations + 1):
                logger.debug("Iteration %s/%s...", it, self.max_iterations)

                cost, trans_delta, angle, rel_par, corr = (
                    self.optimization_step(
                        alpha_stiffness, stiffness_precomputed
                    )
                )

                rel_cost = torch.abs(prev_cost - cost) / (prev_cost + 1e-12)
                prev_cost = cost
                costs.append(cost.item())

                # Check convergence
                if (
                    trans_delta < self.trans_tol
                    and angle < self.angle_tol
                    and rel_par < self.param_tol
                    and rel_cost < self.cost_tol
                ):
                    converged = True
                    break

        return {
            "final_cost": prev_cost,
            "costs": costs,
            "converged": converged,
            "last_trans_delta": trans_delta,
            "last_rot_angle": angle,
        }

    def transform(self):
        """Transform the source shape using the computed deformation."""
        # Transform points
        v_h = torch.cat(
            [self.src_orig, torch.ones(self.n, 1, device=self.device)], 1
        )
        final_pts_trans = torch.einsum("nij,nj->ni", self.X_total, v_h)[:, :3]

        # Create transformed shape
        if HAS_PYVISTA and isinstance(self.source_shape, pv.PolyData):
            transformed_shape = self.source_shape.copy()
            transformed_shape.points = final_pts_trans.cpu().numpy()
        elif isinstance(self.source_shape, torch.Tensor | np.ndarray):
            if isinstance(self.source_shape, torch.Tensor):
                transformed_shape = final_pts_trans.to(
                    self.source_shape.device
                )
            else:
                transformed_shape = final_pts_trans.cpu().numpy()
        elif (
            isinstance(self.source_shape, tuple)
            and len(self.source_shape) == 2
        ):
            transformed_points = final_pts_trans
            if self.src_nml_orig is not None:
                # Transform normals
                Rb = self.X_total[:, :3, :3]
                transformed_normals = torch.einsum(
                    "nij,nj->ni", Rb, self.src_nml_orig
                )
            else:
                transformed_normals = self.source_shape[1]

            if isinstance(self.source_shape[0], torch.Tensor):
                transformed_points = transformed_points.to(
                    self.source_shape[0].device
                )
                if isinstance(
                    self.source_shape[1], torch.Tensor
                ) and isinstance(transformed_normals, torch.Tensor):
                    transformed_normals = transformed_normals.to(
                        self.source_shape[1].device
                    )

            transformed_shape = (transformed_points, transformed_normals)
        else:
            transformed_shape = final_pts_trans.cpu().numpy()

        return transformed_shape, final_pts_trans


def non_rigid_icp(
    source_shape: torch.Tensor | Any,
    target_shape: torch.Tensor | Any,
    robust_loss: str = "cauchy_mad",
    metric_type: str = "point_to_point",
    max_iterations: int = 50,
    device: torch.device | None = None,
    gamma=1.0,
    alpha_start: float = 1e5,
    alpha_final: float = 1e1,
    n_levels: int = 6,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    """
    Non Rigid ICP using the NonRigidDeformation class.
    """
    tgt_pts, _ = _extract_points_and_normals(target_shape)
    if device is not None:
        tgt_pts = tgt_pts.to(device)

    coupling = ClosestPointCoupling(device=device)
    coupling.initialize(tgt_pts, leaf_size=40)

    model = NonRigidDeformation(
        robust_loss=robust_loss,
        metric_type=metric_type,
        max_iterations=max_iterations,
        gamma=gamma,
        device=device,
        coupling=coupling,
        alpha_start=alpha_start,
        alpha_final=alpha_final,
        n_levels=n_levels,
        **kwargs,
    )

    model.initialize(source_shape, target_shape)
    info = model.fit()
    transformed_shape, final_pts_trans = model.transform()

    final_corr = model._compute_correspondences(final_pts_trans)

    return model.X_total, info, final_pts_trans, transformed_shape, final_corr
