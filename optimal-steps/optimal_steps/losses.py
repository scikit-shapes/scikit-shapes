"""Losses (Section 2 of the paper, "Loss Functions").

Each loss provides, for the 3x3 blocks L_i of the paper:
- apply_L(v):              L v, without the landmark term,
- apply_L_inv(v):          L_eff^-1 v, with L_eff = L + landmark weights,
- preconditioner(d, lam):  v -> P^-1 v, block-Jacobi preconditioner of
                           (K~ + lam L_eff^-1), where d is the diagonal of K~.
`landmark_weights` is 0.0 or an (N, 1) tensor.
"""

import torch


class PointToPointLoss:
    """L_i = I: squared Euclidean distance."""

    def __init__(self, alpha, beta, target_normals, source_normals, landmark_weights):
        self.inv_L_eff = 1.0 / (1.0 + landmark_weights)

    def apply_L(self, v):
        return v

    def apply_L_inv(self, v):
        return v * self.inv_L_eff

    def preconditioner(self, diag_K, lambda_reg):
        inv_prec = 1.0 / (diag_K + lambda_reg * self.inv_L_eff)
        return lambda v: v * inv_prec


class PointToPlaneLoss:
    """L_i = beta I + (alpha - beta) m m^T, with m the normal of the matched target point."""

    def __init__(self, alpha, beta, target_normals, source_normals, landmark_weights):
        self.alpha, self.beta, self.n = alpha, beta, target_normals
        self.inv_alpha = 1.0 / (alpha + landmark_weights)
        self.inv_beta = 1.0 / (beta + landmark_weights)

    def _split(self, v, along_normal, orthogonal):
        # along_normal on the normal direction, orthogonal on the tangent plane
        v_in = v.view(-1, 3)
        dot = (v_in * self.n).sum(1, keepdim=True)
        return (orthogonal * v_in + (along_normal - orthogonal) * dot * self.n).view_as(v)

    def apply_L(self, v):
        return self._split(v, self.alpha, self.beta)

    def apply_L_inv(self, v):
        return self._split(v, self.inv_alpha, self.inv_beta)

    def preconditioner(self, diag_K, lambda_reg):
        inv_mu_alpha = 1.0 / (diag_K + lambda_reg * self.inv_alpha + 1e-12)
        inv_mu_beta = 1.0 / (diag_K + lambda_reg * self.inv_beta + 1e-12)
        return lambda v: self._split(v, inv_mu_alpha, inv_mu_beta)


class PlaneToPlaneLoss:
    """L_i = 2 beta I + (alpha - beta) (n n^T + m m^T), with n the source normal and m the
    normal of the matched target point (Eq. plane_to_plane of the paper, alpha = 1)."""

    def __init__(self, alpha, beta, target_normals, source_normals, landmark_weights):
        self.alpha, self.beta = alpha, beta
        self.n_t = target_normals
        self.n_s = torch.nn.functional.normalize(source_normals, dim=1)
        # Eigenvectors of L_i and their eigenvalues (plus the landmark weight)
        self.u = (
            torch.cross(self.n_s, self.n_t, dim=1),
            self.n_s + self.n_t,
            self.n_s - self.n_t,
        )
        self.u_sq = tuple((u**2).sum(1, keepdim=True) for u in self.u)
        rho = (self.n_t * self.n_s).sum(1, keepdim=True)
        self.eigenvalues = (
            2.0 * beta + landmark_weights,
            alpha + beta + landmark_weights + (alpha - beta) * rho,
            alpha + beta + landmark_weights - (alpha - beta) * rho,
        )

    def _spectral(self, v, factors):
        # sum_k factor_k <v, u_k> u_k / |u_k|^2
        v_in = v.view(-1, 3)
        terms = [
            f * ((v_in * u).sum(1, keepdim=True) / (u_sq + 1e-12)) * u
            for f, u, u_sq in zip(factors, self.u, self.u_sq)
        ]
        return (terms[0] + terms[1] + terms[2]).view_as(v)

    def apply_L(self, v):
        v_in = v.view(-1, 3)
        dot_t = (v_in * self.n_t).sum(1, keepdim=True)
        dot_s = (v_in * self.n_s).sum(1, keepdim=True)
        return (
            2 * self.beta * v_in + (self.alpha - self.beta) * (dot_t * self.n_t + dot_s * self.n_s)
        ).view_as(v)

    def apply_L_inv(self, v):
        return self._spectral(v, [1.0 / (lam + 1e-12) for lam in self.eigenvalues])

    def preconditioner(self, diag_K, lambda_reg):
        factors = [1.0 / (diag_K + lambda_reg / (lam + 1e-12) + 1e-12) for lam in self.eigenvalues]
        return lambda v: self._spectral(v, factors)


LOSSES = {
    "point2point": PointToPointLoss,
    "point2plane": PointToPlaneLoss,
    "plane2plane": PlaneToPlaneLoss,
}
