"""Optimal Steps for Fast Diffeomorphic Shape Registration.

`register(source, target, config)` pre-aligns the source onto the target (RANSAC on FPFH
features, ICP, anisotropic scaling), then deforms it with the diffeomorphic registration
of the paper (DiffeomorphicRegistration), coarse to fine. RegistrationConfig holds the
parameters.

Modules:
    config     RegistrationConfig, device selection
    geometry   mesh utilities: vertex areas, edge lengths
    io         loading of meshes, point clouds and segmentations
    alignment  rigid + anisotropic pre-alignment
    matching   FPFH descriptors, target positions z_i
    losses     point-to-point, point-to-plane and plane-to-plane losses
    solver     preconditioned conjugate gradient
    model      DiffeomorphicRegistration
    api        register(), RegistrationResult
    metrics    Chamfer distance, HD95, log-Jacobian variance (Table 1)
    cli        command-line interface
"""

# Must stay the first import: sets up the compiler KeOps uses on macOS, before pykeops is imported.
from . import _keops_setup  # noqa: F401  # isort: skip

from .alignment import align_rigid
from .api import RegistrationResult, register
from .config import LAMBDA_REG_REFERENCE_POINTS, RegistrationConfig, per_scale, resolve_device
from .geometry import compute_vertex_areas, get_average_edge_length
from .io import load_input
from .losses import LOSSES, PlaneToPlaneLoss, PointToPlaneLoss, PointToPointLoss
from .matching import compute_fpfh, effective_targets
from .model import DiffeomorphicRegistration
from .solver import cg

__all__ = [
    "DiffeomorphicRegistration",
    "LAMBDA_REG_REFERENCE_POINTS",
    "LOSSES",
    "PlaneToPlaneLoss",
    "PointToPlaneLoss",
    "PointToPointLoss",
    "RegistrationConfig",
    "RegistrationResult",
    "align_rigid",
    "cg",
    "compute_fpfh",
    "compute_vertex_areas",
    "effective_targets",
    "get_average_edge_length",
    "load_input",
    "per_scale",
    "register",
    "resolve_device",
]
