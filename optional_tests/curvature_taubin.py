import torch
torch.set_printoptions(precision=3, sci_mode=False)

b = 1000
n = 100

X = torch.randn(b, n, 3).float()
X[..., 2] = X[...,0] ** 2 + X[...,1] ** 2 + 3 * X[...,0] + 2

x, y, z = X[..., 0], X[..., 1], X[..., 2]
ones = torch.ones_like(x)
zero = torch.zeros_like(x)

L = torch.stack([x**2, y**2, z**2, x*y, x*z, y*z, x, y, z, ones], dim=-1)
Lx = torch.stack([2*x, zero, zero, y, z, zero, ones, zero, zero, zero], dim=-1)
Ly = torch.stack([zero, 2*y, zero, x, zero, z, zero, ones, zero, zero], dim=-1)
Lz = torch.stack([zero, zero, 2*z, zero, x, y, zero, zero, ones, zero], dim=-1)
assert L.shape == (b, n, 10)
assert Lx.shape == (b, n, 10)
assert Ly.shape == (b, n, 10)
assert Lz.shape == (b, n, 10)


M = (L.transpose(1,2) @ L) / n
print(M.shape)

N = (Lx.transpose(1,2) @ Lx + Ly.transpose(1,2) @ Ly + Lz.transpose(1,2) @ Lz) / n 

import time
start = time.time()
eigenvalues, eigenvectors = torch.lobpcg(A=M, B=N, k=1, largest=False, tol=1e-5)
print('time', time.time() - start)

eigenvectors = eigenvectors.view(b, -1)
print(eigenvalues)
print((eigenvectors / eigenvectors[:,0].view(-1, 1)))
print((L @ eigenvectors.view(b, -1, 1)).abs().mean(dim=1).max())

