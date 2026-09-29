"""The diffeomorphic registration of the paper: matching, regularisation, deformation."""

import logging
from typing import Optional, Union

import numpy as np
import pyvista as pv
import torch
from pykeops.torch import LazyTensor

from .config import LAMBDA_REG_REFERENCE_POINTS, RegistrationConfig, per_scale, resolve_device
from .geometry import compute_vertex_areas, get_average_edge_length
from .losses import LOSSES
from .matching import compute_fpfh, effective_targets
from .solver import cg


class DiffeomorphicRegistration:
    """Diffeomorphic registration of a source onto a target (Section 2 of the paper).

    run() loops over the scales (kernel radius from sigma_init to sigma_final) and, at
    each of the outer_steps steps of a scale:
    1. matching: compute_correspondences(), then effective_targets() gives z_i,
    2. regularisation: solve_gauss_newton() gives the momentum p of the smooth displacement,
    3. deformation: step_euler() moves the source along the geodesic of momentum p.

    Subclasses can redefine compute_correspondences() to match other data (see
    register_lungs.VesselTreeRegistration).
    """

    def __init__(
        self,
        source_mesh: pv.PolyData,
        target_mesh: pv.PolyData,
        config: RegistrationConfig,
        source_features: Optional[Union[np.ndarray, torch.Tensor]] = None,
        target_features: Optional[Union[np.ndarray, torch.Tensor]] = None,
        source_landmark_indices=None,
        target_landmark_indices=None,
    ):
        self.config = config
        self.device = resolve_device(config.device)

        self.source_mesh_template = source_mesh.copy()
        self.m = compute_vertex_areas(source_mesh).to(self.device).view(-1, 1)
        self.q0 = torch.tensor(source_mesh.points, dtype=torch.float32, device=self.device)
        self.tgt = torch.tensor(target_mesh.points, dtype=torch.float32, device=self.device)
        self.N = self.q0.shape[0]

        self._init_normals(source_mesh, target_mesh)

        n_scales = len(config.sigmas) if config.sigmas is not None else int(config.n_scales)
        if n_scales < 1:
            raise ValueError("At least one sigma value is required.")

        # Feature weights are relative to the size of the source.
        mesh_size = source_mesh.length
        self.use_fpfh = config.use_fpfh
        self.fpfh_radius = config.fpfh_radius
        self.normal_weight = float(config.normal_weight) * mesh_size
        self.fpfh_weights = per_scale(config.fpfh_weight, n_scales, "fpfh_weight", mesh_size)
        self.feature_weights = per_scale(config.feature_weight, n_scales, "feature_weight", mesh_size)
        self.current_fpfh_weight = self.fpfh_weights[0]
        self.current_feature_weight = self.feature_weights[0]

        self.src_custom_features, self.tgt_custom_features = self._prepare_custom_features(
            source_features, target_features
        )
        if self.src_custom_features is not None:
            print(
                f"Custom features on: {self.src_custom_features.shape[1]} channel(s), "
                f"effective weights: {[round(w, 4) for w in self.feature_weights]}"
            )
        self.tgt_fpfh = (
            compute_fpfh(self.tgt, self.tgt_normals, self.fpfh_radius) if self.use_fpfh else None
        )
        self.tgt_features = self._compose_features(
            self.tgt, self.tgt_normals, self.tgt_fpfh, self.tgt_custom_features
        )

        self.use_symmetric_correspondences = config.use_symmetric_correspondences
        self.trust_symmetric_values = per_scale(config.trust_symmetric, n_scales, "trust_symmetric")
        self.trust_symmetric = self.trust_symmetric_values[0]

        mean_dist = get_average_edge_length(source_mesh)
        self._init_sigmas(n_scales, mean_dist)

        # lambda_reg is rescaled by the number of source points (see LAMBDA_REG_REFERENCE_POINTS).
        self.lambda_regs = per_scale(
            config.lambda_reg, n_scales, "lambda_reg", self.N / LAMBDA_REG_REFERENCE_POINTS
        )
        self.lambda_reg = self.lambda_regs[0]

        self.metric_type = config.metric_type
        if self.metric_type not in LOSSES:
            raise ValueError(f"Unknown metric type: {self.metric_type}")
        self.metric_alpha = config.metric_alpha
        self.metric_beta = config.metric_beta
        self.solver_precision_mm = config.solver_precision_mm

        self.use_sinkhorn = config.use_sinkhorn
        self.sinkhorn_its = 1
        self.outer_steps = config.outer_steps
        self.euler_precision_step_mm = config.euler_precision_step_mm

        self._init_incompressibility(n_scales, mean_dist)
        self._init_landmarks(source_landmark_indices, target_landmark_indices)

    def _init_normals(self, source_mesh, target_mesh):
        """Unit normals of the source and target points (zeros when unused)."""
        if float(self.config.normal_weight) == 0.0 and not self.config.use_fpfh:
            self.src_normals_orig = torch.zeros((self.N, 3), dtype=torch.float32, device=self.device)
            self.tgt_normals = torch.zeros((self.tgt.shape[0], 3), dtype=torch.float32, device=self.device)
            return

        def unit_normals(mesh):
            if "Normals" not in mesh.point_data:
                mesh.compute_normals(inplace=True)
            normals = torch.tensor(mesh.point_data["Normals"], dtype=torch.float32, device=self.device)
            return torch.nn.functional.normalize(normals, p=2, dim=1)

        self.src_normals_orig = unit_normals(source_mesh)
        self.tgt_normals = unit_normals(target_mesh)

    def _init_sigmas(self, n_scales, mean_dist):
        """Kernel radius of every scale: explicit list, or log-spaced from sigma_init to sigma_final."""
        if self.config.sigmas is not None:
            self.sigmas = np.asarray(self.config.sigmas, dtype=np.float32)
            if np.any(self.sigmas <= 0):
                raise ValueError("All values in sigmas must be strictly positive.")
            print(
                f"Average edge length of source mesh: {mean_dist:.4f}. "
                f"Using explicit sigma schedule: {self.sigmas.tolist()}"
            )
        else:
            print(
                f"Average edge length of source mesh: {mean_dist:.4f}. "
                f"Sigma final: {self.config.sigma_final}, init: {self.config.sigma_init}"
            )
            self.sigmas = np.logspace(
                np.log10(self.config.sigma_init), np.log10(self.config.sigma_final), n_scales
            )
        self.n_scales = len(self.sigmas)

    def _init_incompressibility(self, n_scales, mean_dist):
        """Optional penalty on changes of local density (not used in the paper)."""
        self.incompressibility_weights = per_scale(
            self.config.incompressibility_weight, n_scales, "incompressibility_weight"
        )
        self.incompressibility_weight = self.incompressibility_weights[0]
        self.incompressibility_radius = self.config.incompressibility_radius
        if self.incompressibility_radius is None:
            self.incompressibility_radius = float(mean_dist * 2.5)

        self._rho0 = None
        if any(w > 0 for w in self.incompressibility_weights):
            print(
                f"Incompressibility on with radius: {self.incompressibility_radius:.4f}, "
                f"Weights: {self.incompressibility_weights}"
            )
            with torch.no_grad():
                self._rho0 = self._local_density(self.q0, self.incompressibility_radius).detach()

    def _init_landmarks(self, source_indices, target_indices):
        """Pairs of source/target vertices pulled together by a penalty of weight landmark_weight."""
        if source_indices is None or target_indices is None or len(source_indices) == 0:
            self.lm_idx_src = self.lm_idx_tgt = self.lm_target_pos = self.lm_mask = None
            self.landmark_weight = 0.0
            return

        self.lm_idx_src = torch.as_tensor(source_indices, dtype=torch.long, device=self.device)
        self.lm_idx_tgt = torch.as_tensor(target_indices, dtype=torch.long, device=self.device)
        assert (
            self.lm_idx_src.shape == self.lm_idx_tgt.shape
        ), "Source/target landmark index arrays must have the same length."
        self.lm_target_pos = self.tgt[self.lm_idx_tgt].contiguous()
        self.landmark_weight = float(self.config.landmark_weight)
        self.lm_mask = torch.zeros(self.N, 1, device=self.device)
        self.lm_mask[self.lm_idx_src] = 1.0

    def _prepare_custom_features(self, source_features, target_features):
        # Custom descriptors are optional, but they only make sense as a pair:
        # correspondences are a nearest-neighbour search in one shared feature
        # space, so both sides must carry the same channels.
        if (source_features is None) != (target_features is None):
            raise ValueError(
                "source_features and target_features must be provided together: "
                "correspondences are searched in a single shared feature space."
            )
        if source_features is None:
            return None, None

        def _as_tensor(feat, n_expected, name, mesh_name):
            t = torch.as_tensor(feat, dtype=torch.float32, device=self.device)
            if t.dim() == 1:
                t = t.view(-1, 1)
            if t.dim() != 2:
                raise ValueError(
                    f"{name} must be of shape (n_points, n_channels), "
                    f"got {tuple(t.shape)}."
                )
            if t.shape[0] != n_expected:
                raise ValueError(
                    f"{name} has {t.shape[0]} rows but the {mesh_name} mesh has "
                    f"{n_expected} points. Compute the features on the mesh that is "
                    f"actually passed to the registration: load_input() may clean, "
                    f"triangulate or decimate the input file."
                )
            return t.contiguous()

        src = _as_tensor(source_features, self.N, "source_features", "source")
        tgt = _as_tensor(
            target_features, self.tgt.shape[0], "target_features", "target"
        )
        if src.shape[1] != tgt.shape[1]:
            raise ValueError(
                f"source_features has {src.shape[1]} channels but target_features "
                f"has {tgt.shape[1]}: both must describe the same feature space."
            )
        return src, tgt

    def _compose_features(
        self,
        points: torch.Tensor,
        normals: torch.Tensor,
        fpfh: Optional[torch.Tensor] = None,
        custom: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Single place where a feature vector is assembled, for the source and
        # the target alike: the two sides then carry the same blocks in the same
        # order by construction, which is what the argmin below requires.
        blocks = [points, self.normal_weight * normals]
        if self.use_fpfh:
            if fpfh is None:
                raise ValueError("use_fpfh is set but no FPFH descriptor was given.")
            blocks.append(self.current_fpfh_weight * fpfh)
        if custom is not None:
            blocks.append(self.current_feature_weight * custom)
        return torch.cat(blocks, dim=1).contiguous()

    def _local_density(
        self, q: torch.Tensor, radius: float
    ) -> torch.Tensor:  # for incompressibility, not used in the paper
        r2 = float(radius) ** 2
        x_i = LazyTensor(q[:, None, :])
        y_j = LazyTensor(q[None, :, :])
        D_ij = ((x_i - y_j) ** 2).sum(-1)
        K_ij = (-D_ij / r2).exp()
        m_j = LazyTensor(self.m[None, :, :])
        rho = (K_ij * m_j).sum(dim=1)
        return rho

    def update_source_normals(self, current_points):
        # Point clouds have no polygon cells; skip normal estimation and return zeros.
        if self.source_mesh_template.n_faces_strict == 0:
            return torch.zeros(
                (current_points.shape[0], 3),
                dtype=torch.float32,
                device=self.device,
            )
        points_np = current_points.detach().cpu().numpy()
        self.source_mesh_template.points = points_np
        self.source_mesh_template.compute_normals(inplace=True, flip_normals=False)
        new_normals = torch.tensor(
            self.source_mesh_template.point_data["Normals"],
            dtype=torch.float32,
            device=self.device,
        )
        return torch.nn.functional.normalize(new_normals, p=2, dim=1)

    def get_keops_kernel(self, q, sigma):
        sigma_tensor = torch.tensor([sigma], dtype=torch.float32, device=self.device)
        sigma_lt = LazyTensor(sigma_tensor)
        x_i = LazyTensor(q.detach()[:, None, :])
        y_j = LazyTensor(q[None, :, :])
        D_ij = ((x_i - y_j) ** 2).sum(-1)
        K_ij = (-D_ij.sqrt() / (sigma_lt)).exp()
        return K_ij

    def solve_sinkhorn(self, q, sigma):
        K_ij = self.get_keops_kernel(q, sigma)
        s = torch.ones(self.N, 1, device=self.device)
        for _ in range(self.sinkhorn_its):
            weighted_s = s * self.m
            denom = K_ij @ weighted_s
            s = torch.sqrt(s / (denom + 1e-9))
        return s

    def compute_correspondences(
        self, q, current_src_normals=None, precomputed_src_fpfh=None
    ):
        src_n = (
            current_src_normals
            if current_src_normals is not None
            else self.src_normals_orig
        )

        src_fpfh = None
        if self.use_fpfh:
            src_fpfh = (
                precomputed_src_fpfh
                if precomputed_src_fpfh is not None
                else compute_fpfh(q, src_n, self.fpfh_radius)
            )
        src_features = self._compose_features(
            q, src_n, src_fpfh, self.src_custom_features
        )

        x_i = LazyTensor(src_features[:, None, :])
        y_j = LazyTensor(self.tgt_features[None, :, :])
        d_ij = ((x_i - y_j) ** 2).sum(-1)

        indices_s_to_t = d_ij.argmin(dim=1).long().view(-1)
        y = self.tgt[indices_s_to_t]
        n_y = self.tgt_normals[indices_s_to_t]

        indices_t_to_s = None
        if self.use_symmetric_correspondences:
            indices_t_to_s = d_ij.argmin(dim=0).long().view(-1)

        return y, n_y, indices_t_to_s

    def kernel_operator(self, q, sigma):
        """Normalised kernel of the paper, K~ = S K S (Eq. normalized_kernel).

        Returns v -> K~ v and the diagonal of K~ (s_i^2). Without Sinkhorn normalisation,
        K~ = K.
        """
        K_ij = self.get_keops_kernel(q, sigma)
        if not self.use_sinkhorn:
            return (lambda v: K_ij @ v), torch.ones(self.N, 1, device=self.device)
        s = self.solve_sinkhorn(q, sigma)
        return (lambda v: (K_ij @ (v * s)) * s), (s**2).view(-1, 1)

    def solve_gauss_newton(self, q_k, target_points, n_t, sigma, src_normals=None, x0=None):
        """Momentum p of the regularised displacement: (K~ + lambda L_eff^-1) p = rhs."""
        with torch.no_grad():
            apply_K, diag_K = self.kernel_operator(q_k, sigma)

        has_lm = (self.lm_idx_src is not None) and (self.landmark_weight > 0.0)
        landmark_weights = self.landmark_weight * self.lm_mask if has_lm else 0.0
        loss = LOSSES[self.metric_type](
            self.metric_alpha,
            self.metric_beta,
            n_t,
            src_normals if src_normals is not None else self.src_normals_orig,
            landmark_weights,
        )

        def system_operator(p):
            return apply_K(p) + self.lambda_reg * loss.apply_L_inv(p)

        # Right-hand side: L_eff^-1 [L (z - q) + c_lm S^T (z_lm - S q)] with landmarks,
        # z - q without.
        if has_lm:
            force = loss.apply_L(target_points - q_k).clone()
            z_minus_q_lm = self.lm_target_pos - q_k[self.lm_idx_src]
            force[self.lm_idx_src] = force[self.lm_idx_src] + self.landmark_weight * z_minus_q_lm
            b = loss.apply_L_inv(force)
        else:
            b = target_points - q_k

        dynamic_point_tol = max(self.solver_precision_mm, sigma / 10)
        effective_tol = dynamic_point_tol * np.sqrt(self.N)
        with torch.no_grad():
            p = cg(
                system_operator,
                b,
                P=loss.preconditioner(diag_K, self.lambda_reg),
                max_iter=200,
                tol=effective_tol,
                x0=x0,
            )
        return p.float()

    def step_euler(self, q, p, return_path=False):
        """Geodesic shooting from q with momentum p, adaptive Euler integration."""
        curr_sigma = self.current_sigma

        def get_velocity_and_force(q_curr, p_curr):
            with torch.enable_grad():
                q_in = q_curr.detach().requires_grad_(True)
                p_in = p_curr.detach()
                apply_K, _ = self.kernel_operator(q_in, curr_sigma)
                v = apply_K(p_in)
                H = 0.5 * torch.sum(p_in * v)
                if self.incompressibility_weight > 0 and (self._rho0 is not None):
                    rho = self._local_density(q_in, self.incompressibility_radius)
                    E_inc = torch.mean((rho - self._rho0) ** 2)
                    H = H + (self.incompressibility_weight * E_inc)
                grads = 2 * torch.autograd.grad(H, (q_in), create_graph=False)
                force = -grads[0]
                return v, force

        curr_t = 0.0
        end_t = 1.0
        n_euler_steps_max = 100

        curr_q = q.clone()
        curr_p = p.clone()

        path_q = []
        path_p = []
        if return_path:
            path_q.append(curr_q.clone())
            path_p.append(curr_p.clone())

        n_steps_euler = 0

        while curr_t < end_t:
            v, f = get_velocity_and_force(curr_q, curr_p)
            v_norm = v.norm(p=2, dim=1).max().item()
            if v_norm < 1e-6:
                dt = end_t - curr_t
            else:
                safe_dt = self.euler_precision_step_mm / v_norm
                dt = max(safe_dt, 1 / n_euler_steps_max)
            if curr_t + dt > end_t:
                dt = end_t - curr_t
            curr_q = curr_q + dt * v
            curr_p = curr_p + dt * f
            curr_t += dt
            if return_path:
                path_q.append(curr_q.clone())
                path_p.append(curr_p.clone())
            n_steps_euler += 1

        logging.getLogger(__name__).info(
            f"Euler integration completed in {n_steps_euler} steps (sigma={float(curr_sigma):.4f})"
        )

        if return_path:
            return curr_q, torch.stack(path_q), torch.stack(path_p)
        return curr_q

    def _set_scale(self, scale_idx):
        """Parameters of the current scale."""
        self.current_sigma = self.sigmas[scale_idx]
        self.current_fpfh_weight = self.fpfh_weights[scale_idx]
        self.current_feature_weight = self.feature_weights[scale_idx]
        self.incompressibility_weight = self.incompressibility_weights[scale_idx]
        self.lambda_reg = self.lambda_regs[scale_idx]
        self.trust_symmetric = self.trust_symmetric_values[scale_idx]

    def run(self, return_history=False):
        q = self.q0.clone()
        curr_src_normals = self.src_normals_orig.clone()
        p_cached = torch.zeros_like(q)

        history_q = []
        history_p = []
        if return_history:
            history_q.append(q[None, ...].detach())
            history_p.append(torch.zeros_like(q[None, ...]))

        for scale_idx, sigma in enumerate(self.sigmas):
            self._set_scale(scale_idx)

            # Rebuilt at every scale: the FPFH and custom-feature weights change.
            self.tgt_features = self._compose_features(
                self.tgt, self.tgt_normals, self.tgt_fpfh, self.tgt_custom_features
            )

            cached_src_fpfh = None
            if self.use_fpfh:
                cached_src_fpfh = compute_fpfh(q, curr_src_normals, self.fpfh_radius)

            print(
                f"--- Scale {scale_idx+1}/{len(self.sigmas)}: sigma={sigma:.4f}, "
                f"FPFH_weight={self.current_fpfh_weight:.4f}, "
                f"incompressibility_weight={self.incompressibility_weight:.4f}, "
                f"lambda_reg={self.lambda_reg:.6f}, "
                f"symmetric_matching_weight={self.trust_symmetric:.4f} ---"
            )

            for _ in range(self.outer_steps):
                # 1. Matching
                forward_targets, target_normals, backward_matches = self.compute_correspondences(
                    q, curr_src_normals, precomputed_src_fpfh=cached_src_fpfh
                )
                if not self.use_symmetric_correspondences:
                    backward_matches = None
                z = effective_targets(q, forward_targets, self.tgt, backward_matches, self.trust_symmetric)

                # 2. Regularisation
                p0 = self.solve_gauss_newton(
                    q_k=q,
                    target_points=z,
                    n_t=target_normals,
                    sigma=sigma,
                    src_normals=curr_src_normals,
                    x0=p_cached,
                )
                p_cached = p0.clone()
                if p0.abs().max() > 1e5:
                    p0 = torch.clamp(p0, -1e5, 1e5)

                # 3. Deformation
                q = self.step_euler(q, p0, return_path=False)
                curr_src_normals = self.update_source_normals(q)

                if return_history:
                    history_q.append(q[None, ...].detach())
                    history_p.append(p0[None, ...].detach())

        if return_history:
            full_traj_q = torch.cat(history_q, dim=0)
            full_traj_p = torch.cat(history_p, dim=0)
            return q.detach(), full_traj_q, full_traj_p

        return q.detach()
