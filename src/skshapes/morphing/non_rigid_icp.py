import logging
from typing import Any

import numpy as np
import pyvista as pv
import scipy.sparse as sp
import torch
from sklearn.neighbors import KDTree
from sksparse.cholmod import cholesky

torch.set_default_dtype(torch.float32)
logger = logging.getLogger(__name__)


@torch.no_grad()
def _torch_sparse_to_scipy_csr(t: torch.Tensor) -> sp.csr_matrix:
    t = t.to_sparse_coo().coalesce()
    i = t.indices()
    v = t.values().detach().cpu().numpy()
    r = i[0].detach().cpu().numpy()
    c = i[1].detach().cpu().numpy()
    return sp.coo_matrix((v, (r, c)), shape=t.shape).tocsr()


@torch.no_grad()
def _cotangent_laplacian(
    mesh: pv.PolyData,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-12,
) -> torch.Tensor:

    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()

    V_np = mesh.points  # numpy (N,3)
    N = V_np.shape[0]
    V = torch.as_tensor(V_np, dtype=dtype, device=device)

    F_np = mesh.faces.reshape(-1, 4)[:, 1:4].astype("int64")  # numpy (M,3)
    if F_np.size == 0:
        msg = "Mesh has no faces to compute cotangent Laplacian."
        raise ValueError(msg)
    F = torch.as_tensor(F_np, dtype=torch.long, device=device)  # torch (M,3)

    i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]
    p0, p1, p2 = V[i0], V[i1], V[i2]

    # filter out degenerate triangles (repeated vertices)
    nondeg = (i0 != i1) & (i1 != i2) & (i2 != i0)
    if not torch.all(nondeg):
        i0, i1, i2 = i0[nondeg], i1[nondeg], i2[nondeg]
        p0, p1, p2 = p0[nondeg], p1[nondeg], p2[nondeg]

    # helper: cot(angle at A) for triangle (A,B,C) using vectors AB, AC
    def cot_at(a, b, c):
        u = b - a
        v = c - a
        num = (u * v).sum(dim=1)
        den = torch.linalg.norm(torch.cross(u, v, dim=1), dim=1).clamp_min(eps)
        return num / den

    cot0 = cot_at(p0, p1, p2)  # opposite edge (i1,i2)
    cot1 = cot_at(p1, p2, p0)  # opposite edge (i2,i0)
    cot2 = cot_at(p2, p0, p1)  # opposite edge (i0,i1)

    # each triangle contributes 0.5*cot(angle_at_vertex) to the opposite edge weight
    w0 = 0.5 * cot0
    w1 = 0.5 * cot1
    w2 = 0.5 * cot2

    rows = torch.cat([i1, i2, i2, i0, i0, i1], dim=0)
    cols = torch.cat([i2, i1, i0, i2, i1, i0], dim=0)
    vals = torch.cat([w0, w0, w1, w1, w2, w2], dim=0).to(dtype=dtype)

    # remove any NaN/Inf (can happen if triangles are nearly degenerate)
    finite_mask = torch.isfinite(vals)
    rows, cols, vals = rows[finite_mask], cols[finite_mask], vals[finite_mask]

    # build W (symmetric, off-diagonal)
    W = torch.sparse_coo_tensor(
        indices=torch.stack([rows, cols], dim=0),
        values=vals,
        size=(N, N),
        dtype=dtype,
        device=device,
    ).coalesce()  # torch (N, N)

    # degree vector d = sum_j W_ij
    d = torch.sparse.sum(W, dim=1).to_dense().flatten()

    # diagonal matrix D as sparse
    diag_idx = torch.arange(N, device=device, dtype=torch.long)
    D = torch.sparse_coo_tensor(
        indices=torch.stack([diag_idx, diag_idx], dim=0),
        values=d,
        size=(N, N),
        dtype=dtype,
        device=device,
    ).coalesce()  # torch (N, N)

    # cotangent Laplacian: L = D - W
    return torch.sparse_coo_tensor(
        indices=torch.cat([D.indices(), W.indices()], dim=1),
        values=torch.cat([D.values(), -W.values()]),
        size=(N, N),
        dtype=dtype,
        device=device,
    ).coalesce()  # torch (N, N)


def _extract_points_and_normals(
    shape: pv.PolyData | torch.Tensor | np.ndarray | tuple,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Extract points and normals from various input formats."""
    if isinstance(shape, pv.PolyData):
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


def _robust_loss_weights(
    residuals: torch.Tensor, loss_type: str = "cauchy_mad", **kwargs
) -> torch.Tensor:
    eps = kwargs.get("eps", 1e-8)

    if loss_type == "l1":
        return 1.0 / torch.sqrt(torch.abs(residuals) + eps)

    elif loss_type == "var_trim":
        trim_ratio = kwargs.get("trim_ratio", 0.8)
        threshold = torch.quantile(residuals, trim_ratio)
        return (residuals <= threshold).to(dtype=residuals.dtype)

    elif loss_type == "cauchy":
        k = kwargs.get("k", 1.0)
        return 1.0 / torch.sqrt(1.0 + (residuals / k) ** 2)

    elif loss_type == "cauchy_mad":
        median_res = torch.median(residuals)
        mad = torch.median(torch.abs(residuals - median_res))
        s = 1.4826 * mad
        k = kwargs.get("k", 1.0)
        normalized_res = residuals / (s + eps)
        return 1.0 / torch.sqrt(1.0 + (normalized_res / k) ** 2)

    elif loss_type == "welsch":
        k = kwargs.get("k", 1.0)
        return torch.exp(-0.5 * (residuals / k) ** 2)

    else:
        return torch.ones_like(residuals)  # torch (N,)


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

    return cov  # torch (N, 3, 3)


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
        landmarks: tuple[torch.Tensor, torch.Tensor] | None = None,
        beta_landmarks: float = 0.0,
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

        self.beta_landmarks = float(beta_landmarks)
        self.landmarks = (
            landmarks  # (idx: Long[N_l], pos: Tensor[N_l,3]) in target coords
        )

        self.coupling = coupling

    def initialize(self, source_shape: Any, target_shape: Any):
        """Initialize the deformation with source and target shapes."""
        self.source_shape = source_shape
        self.target_shape = target_shape
        L_torch = _cotangent_laplacian(self.source_shape)  # torch (N,N)
        L_s = _torch_sparse_to_scipy_csr(L_torch).astype(
            np.float64
        )  # scipy (N,N)
        d = np.array([1.0, 1.0, 1.0, float(self.gamma)] * 3, dtype=np.float64)
        D12 = sp.diags(d)  # scipy sparse (12,12)
        self.R_s = sp.kron(L_s, D12, format="csc")  # scipy sparse (12N,12N)
        self.R_s.sort_indices()

        self.src_pts, self.src_nml = _extract_points_and_normals(
            source_shape
        )  # torch (N, 3), torch (N, 3)
        self.tgt_pts, self.tgt_nml = _extract_points_and_normals(
            target_shape
        )  # torch (M, 3), torch (M, 3)

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

        if self.landmarks is not None and self.beta_landmarks > 0.0:
            idx, pos = self.landmarks
            if isinstance(idx, np.ndarray):
                idx = torch.from_numpy(idx)
            if isinstance(pos, np.ndarray):
                pos = torch.from_numpy(pos).float()
            self.lm_idx = idx.to(self.device).long()
            self.lm_pos = pos.to(self.device).float()
        else:
            self.lm_idx = None
            self.lm_pos = None

        if (
            self.metric_type in ["point_to_plane", "plane_to_plane"]
            and self.tgt_nml is None
        ):
            msg = f"Target normals are required for metric_type='{self.metric_type}' but not provided"
            raise ValueError(msg)
        if self.metric_type == "plane_to_plane" and self.src_nml is None:
            msg = "Source normals are required for metric_type='plane_to_plane' but not provided"
            raise ValueError(msg)

        self.N = self.src_orig.shape[0]
        self.d = 12

        # precompute the torch motifs
        i = torch.arange(self.N, device=self.device)  # torch (N,)
        r = torch.arange(3, device=self.device)  # torch (3,)
        k = torch.arange(4, device=self.device)  # torch (4,)
        Idx, R, K = torch.meshgrid(i, r, k, indexing="ij")

        self._I = Idx
        self._K = K

        # precompute the numpy motif of M
        self._M_rows_np = (
            (3 * Idx + R).reshape(-1).detach().cpu().numpy().astype(np.int64)
        )  # numpy (12N,)
        self._M_cols_np = (
            (12 * Idx + 4 * R + K)
            .reshape(-1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )  # numpy (12N,)

        # precompute the scipy motifs
        m, n = 3 * self.N, 12 * self.N
        self._M_pattern = sp.coo_matrix(
            (
                np.zeros_like(self._M_rows_np, dtype=np.float64),
                (self._M_rows_np, self._M_cols_np),
            ),
            shape=(m, n),
        ).tocsr()  # scipy sparse (3N, 12N)

        indptr = np.arange(self.N + 1, dtype=np.int32)  # numpy (N+1,)
        indices = np.arange(self.N, dtype=np.int32)  # numpy (N,)
        data = np.zeros((self.N, 3, 3), dtype=np.float64)  # numpy (N, 3, 3)
        self._L_bsr = sp.bsr_matrix(
            (data, indices, indptr), shape=(3 * self.N, 3 * self.N)
        )  # scipy sparse (3N, 3N)

        self.X_total = (
            torch.eye(4, device=self.device).unsqueeze(0).repeat(self.N, 1, 1)
        )  # torch (N, 4, 4)

        if self.coupling is None:
            self.tree = KDTree(
                self.tgt.cpu().numpy(), leaf_size=40, metric="euclidean"
            )
        else:
            self.tree = None

        if self.metric_type == "local_covariance":
            k_neighbors = self.kwargs.get("k_neighbors", 20)
            self.cov_src = _compute_local_covariance(
                self.src, k_neighbors
            )  # torch (N, 3, 3)
            self.cov_tgt = _compute_local_covariance(
                self.tgt, k_neighbors
            )  # torch (M, 3, 3)
        else:
            self.cov_src = self.cov_tgt = None

        self._setup_edges()

        if "k" not in self.kwargs:
            s = np.median(
                KDTree(self.src_orig.cpu().numpy()).query(
                    self.src_orig.cpu().numpy(), k=2
                )[0][:, 1]
            )
            alpha = float(self.kwargs.get("alpha", 100.0))
            beta = float(self.kwargs.get("beta", 1.0))
            lambda_bar = (2.0 * alpha + 2.0 * beta) / 3.0
            k_est = 3.0 * s * np.sqrt(lambda_bar)
            self.kwargs["k"] = float(k_est)

        self.alpha_levels = torch.logspace(
            torch.log10(torch.tensor(self.alpha_start)),
            torch.log10(torch.tensor(self.alpha_final)),
            steps=self.n_levels,
            device=self.device,
        )

    def _setup_edges(self):
        """Extract edges from the source mesh for stiffness regularization."""
        if hasattr(self.source_shape, "extract_all_edges"):
            edge_mesh = self.source_shape.extract_all_edges()
            lines = edge_mesh.lines.reshape(-1, 3)
            edges_np = lines[:, 1:]

            edges_unique = np.unique(np.sort(edges_np, axis=1), axis=0)
            pts = self.src_orig.cpu().numpy()
            v0, v1 = pts[edges_unique[:, 0]], pts[edges_unique[:, 1]]
            lengths = np.linalg.norm(v0 - v1, axis=1, keepdims=True)

            # filter out degenerate edges
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
            msg = "No edge information available"
            raise ValueError(msg)

    def _compute_correspondences(
        self, pts_trans: torch.Tensor
    ) -> torch.Tensor:
        """Find closest point correspondences."""
        if self.coupling is not None:
            # use provided coupling
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
            L = torch.eye(3, device=self.device).repeat(
                self.N, 1, 1
            )  # torch (N, 3, 3)

        elif self.metric_type == "point_to_plane":
            tgt_corr_nml = (
                self.tgt_nml[corr] if self.tgt_nml is not None else None
            )
            tgt_corr = self.tgt[corr]
            nrm = self.tgt_nml[corr]  # torch (N, 3)
            nrm = nrm / (
                torch.norm(nrm, dim=1, keepdim=True) + 1e-12
            )  # torch (N,3)
            nnT = torch.einsum("ni,nj->nij", nrm, nrm)  # torch (N,3,3)
            L = alpha * nnT + beta * (
                torch.eye(3, device=self.device)
                .unsqueeze(0)
                .expand(self.N, 3, 3)
                - nnT
            )  # torch (N, 3, 3)
            diff = pts_trans - tgt_corr
            res2 = torch.einsum("ni,nij,nj->n", diff, L, diff)

        elif self.metric_type == "plane_to_plane":
            Rb = self.X_total[:, :3, :3]
            src_nml_trans = torch.einsum("nij,nj->ni", Rb, self.src_nml_orig)
            tgt_corr_nml = self.tgt_nml[corr]

            ns = src_nml_trans / (
                torch.norm(src_nml_trans, dim=1, keepdim=True) + 1e-12
            )  # torch (N, 3)
            nt = tgt_corr_nml / (
                torch.norm(tgt_corr_nml, dim=1, keepdim=True) + 1e-12
            )  # torch (N, 3)

            I3 = (
                torch.eye(3, device=self.device)
                .unsqueeze(0)
                .expand(self.N, 3, 3)
            )
            Ns = torch.einsum("ni,nj->nij", ns, ns)  # torch (N,3,3)
            Nt = torch.einsum("ni,nj->nij", nt, nt)  # torch (N,3,3)
            Ps = I3 - Ns
            Pt = I3 - Nt

            alpha = self.kwargs.get("alpha", 100.0)
            beta = self.kwargs.get("beta", 1.0)

            L = (
                alpha * (Ns + Nt) + 0.5 * beta * (Ps + Pt) + 1e-6 * I3
            )  # torch (N, 3, 3)

            tgt_corr = self.tgt[corr]
            diff = pts_trans - tgt_corr
            res2 = torch.einsum("ni,nij,nj->n", diff, L, diff)

        elif self.metric_type == "local_covariance":
            Rb = self.X_total[:, :3, :3]
            rotcov = torch.bmm(
                Rb, torch.bmm(self.cov_src, Rb.transpose(1, 2))
            )  # torch (N, 3, 3)
            cov_sum = rotcov + self.cov_tgt[corr]  # torch (N, 3, 3)
            L = torch.linalg.inv(cov_sum)  # torch (N, 3, 3)
            tgt_corr = self.tgt[corr]
            diff = pts_trans - tgt_corr
            res2 = torch.einsum("ni,nij,nj->n", diff, L, diff)

        return (
            res2,  # torch (N, 1)
            L,  # torch (3N, 3N)
            diff,  # torch (N, 3)
        )

    def _corr_mask(self, corr, max_angle_deg=60.0):
        if self.tgt_nml is None or self.src_nml_orig is None:
            return torch.ones(self.N, dtype=torch.bool, device=self.device)

        Rb = self.X_total[:, :3, :3]
        ns = torch.einsum("nij,nj->ni", Rb, self.src_nml_orig)
        ns = ns / (torch.norm(ns, dim=1, keepdim=True) + 1e-12)

        nt = self.tgt_nml[corr]
        nt = nt / (torch.norm(nt, dim=1, keepdim=True) + 1e-12)

        cosang = (ns * nt).sum(1).clamp(-1, 1)
        return cosang >= torch.cos(
            torch.tensor(np.deg2rad(max_angle_deg), device=self.device)
        )

    def optimization_step(self, alpha_stiffness: float):
        """Perform one optimization step."""

        v_h = torch.cat(
            [self.src_orig, torch.ones(self.N, 1, device=self.device)], 1
        )  # torch (N, 4)
        pts_trans = torch.einsum("nij,nj->ni", self.X_total, v_h)[
            :, :3
        ]  # torch (N, 3)

        corr = self._compute_correspondences(pts_trans)  # torch (N, 1)
        res2, L, _ = self._compute_metric(
            pts_trans, corr
        )  # torch (N, 1), torch (3N, 3N), torch (N, 3)

        residuals = torch.sqrt(res2)
        weights = _robust_loss_weights(
            residuals, self.robust_loss, **self.kwargs
        ).clamp_min(
            1e-4
        )  # torch (N,)

        mask = self._corr_mask(corr, max_angle_deg=60.0)
        weights = weights * mask.float()

        cost = 0.5 * torch.sum(weights * res2)

        # fill M and L with the actual values
        xh = torch.cat(
            [pts_trans, torch.ones(self.N, 1, device=self.device)], dim=1
        )  # torch (N,4)
        vals = (
            xh[self._I, self._K]
            .reshape(-1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )  # numpy (12N,)

        M_s = self._M_pattern.copy()  # scipy sparse (12N, 4)
        M_s.data[:] = vals

        L_blocks = L.detach().cpu().numpy().astype(np.float64)  # numpy (N,3,3)

        L_bsr = self._L_bsr.copy()  # scipy sparse (3N, 3N)
        L_bsr.data[:] = L_blocks
        L_s = L_bsr.tocsr()
        R_s = self.R_s  # scipy sparse (12N, 12N)

        w_rep = weights.repeat_interleave(3)  # torch (3N,)
        w_rep_np = (
            w_rep.detach().cpu().numpy().astype(np.float64)
        )  # numpy (3N,)
        M_sw = M_s.multiply(w_rep_np[:, None])  # scipy sparse (3N,12N)

        eps_tikh = 1e-8
        A_data = M_sw.T @ (L_s @ M_sw)  # scipy sparse (12N,12N)
        A = (A_data + float(alpha_stiffness) * R_s).tocsr()
        # lines commented because it works fine without, but can improve stability, maybe put condition on sp.linalg.norm(A - A.T)/sp.linalg.norm(A)
        # A.sum_duplicates()
        # A = 0.5 * (A + A.T)
        A = A + eps_tikh * sp.eye(A.shape[0], dtype=A.dtype, format="csr")
        A = A.astype(np.float64).tocsc()
        A.sort_indices()

        y = self.tgt[corr]  # torch (N,3)
        r = (
            (y - pts_trans).flatten().detach().cpu().numpy().astype(np.float64)
        )  # numpy (3N,)
        r_w = w_rep_np * r
        pre_data = 0.5 * float(r_w @ (L_s @ r_w))
        pre_reg = 0.0
        pre_model_cost = pre_data + pre_reg

        b = M_sw.T @ (L_s @ r_w)  # numpy (12N,)
        b = b.astype(np.float64)

        if self.lm_idx is not None and self.beta_landmarks > 0.0:
            beta = float(self.beta_landmarks)

            pts_land = pts_trans[self.lm_idx]  # torch (N_l, 3)
            r_land = (self.lm_pos - pts_land).reshape(-1)  # torch (3*N_l,)
            r_land_np = r_land.detach().cpu().numpy().astype(np.float64)

            lm_idx_np = self.lm_idx.detach().cpu().numpy().astype(np.int64)
            row_ids = (3 * lm_idx_np[:, None] + np.arange(3)[None, :]).reshape(
                -1
            )  # numpy (3*N_l,)
            M_land = M_s[row_ids, :]  # scipy sparse (3*N_l, 12N)

            sqb = np.sqrt(beta)
            M_land_s = M_land * sqb
            r_land_s = sqb * r_land_np

            A = (A + M_land_s.T @ M_land_s).tocsc()
            b = b + M_land_s.T @ r_land_s

        self._chol = cholesky(A, ordering_method="amd")
        delta_flat = self._chol(b)

        e = M_s @ delta_flat - r
        e_w = w_rep_np * e
        post_data = 0.5 * float(e_w @ (L_s @ e_w))
        post_reg = (
            0.5
            * float(alpha_stiffness)
            * float(delta_flat @ (R_s @ delta_flat))
        )
        post_reg += 0.5 * eps_tikh * float(delta_flat @ delta_flat)
        post_model_cost = post_data + post_reg

        Delta = torch.from_numpy(delta_flat.reshape(self.N, 3, 4)).to(
            self.device
        )  # torch (N, 3, 4)
        R_step = (
            torch.eye(3, device=self.device).unsqueeze(0) + Delta[:, :3, :3]
        )  # torch (N, 3, 3), I + Delta

        X_step = torch.zeros(
            self.N, 4, 4, device=self.device
        )  # torch (N, 4, 4)
        X_step[:, :3, :3] = R_step
        X_step[:, :3, 3] = Delta[:, :, 3]
        X_step[:, 3, 3] = 1.0

        X_total_prev = self.X_total.clone()
        self.X_total = torch.bmm(X_step, self.X_total)  # torch (N, 4, 4)

        # convergence checks variables
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
        delta_X = torch.norm((self.X_total - X_total_prev).view(-1))
        rel_par = delta_X / (torch.norm(X_total_prev.view(-1)) + 1e-12)

        return (
            cost,
            trans_delta,
            angle,
            rel_par,
            corr,
            pre_model_cost,
            post_model_cost,
        )

    def fit(self):
        """Run the complete non-rigid ICP optimization."""
        prev_cost = None
        costs = []
        model_costs_pre = []
        model_costs_post = []
        converged = False
        transformed_shapes = []

        for level, alpha_stiffness in enumerate(self.alpha_levels, 1):
            logger.debug(
                "\n=== alpha-level %s/%s : alpha = %.1e ===",
                level,
                self.n_levels,
                alpha_stiffness,
            )

            for it in range(1, self.max_iterations + 1):
                logger.debug("Iteration %s/%s...", it, self.max_iterations)

                (
                    cost,
                    trans_delta,
                    angle,
                    rel_par,
                    corr,
                    pre_model,
                    post_model,
                ) = self.optimization_step(
                    alpha_stiffness=alpha_stiffness,
                )

                costs.append(cost.item())
                model_costs_pre.append(pre_model)
                model_costs_post.append(post_model)

                if prev_cost is None:
                    rel_cost = torch.tensor(float("inf"), device=self.device)
                else:
                    rel_cost = torch.abs(prev_cost - cost) / (
                        prev_cost + 1e-12
                    )
                prev_cost = cost

                if (
                    trans_delta < self.trans_tol
                    and angle < self.angle_tol
                    and (rel_par < self.param_tol or rel_cost < self.cost_tol)
                ):
                    converged = True
                    break

            transformed_shape, _ = self.transform()
            transformed_shapes.append(transformed_shape)

        return {
            "final_cost": prev_cost,
            "costs": costs,
            "model_costs_pre": model_costs_pre,
            "model_costs_post": model_costs_post,
            "converged": converged,
            "last_trans_delta": trans_delta,
            "last_rot_angle": angle,
            "correspondences": corr,
            "transformed_shapes": transformed_shapes,
        }

    def transform(self):
        """Transform the source shape using the computed deformation."""

        v_h = torch.cat(
            [self.src_orig, torch.ones(self.N, 1, device=self.device)], 1
        )
        final_pts_trans = torch.einsum("nij,nj->ni", self.X_total, v_h)[:, :3]

        if isinstance(self.source_shape, pv.PolyData):
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
    final_corr = info.get("correspondences", None)

    return model.X_total, info, final_pts_trans, transformed_shape, final_corr
