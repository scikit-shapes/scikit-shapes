"""Preconditioned conjugate gradient (Section 2 of the paper, "Efficient Linear Solver")."""

import torch


def cg(
    matmul_closure, b, P=None, x0=None, max_iter=100, tol=1e-6
):  # cg with block-diagonal Jacobi preconditioner
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - matmul_closure(x)

    z = P(r) if P is not None else r.clone()
    p = z.clone()
    rs_old = torch.sum(r * z)

    for i in range(max_iter):
        Ap = matmul_closure(p)
        alpha = rs_old / (torch.sum(p * Ap) + 1e-20)
        x = x + alpha * p
        r = r - alpha * Ap

        if torch.norm(r) < tol:
            print(f"CG converged at iteration {i}")
            break

        z = P(r) if P is not None else r
        rs_new = torch.sum(r * z)
        beta = rs_new / (rs_old + 1e-20)
        p = z + beta * p
        rs_old = rs_new

    if i == max_iter - 1:
        print(f"CG reached max_iter ({max_iter}) without full convergence.")
    return x
